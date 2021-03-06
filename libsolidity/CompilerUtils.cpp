/*
	This file is part of cpp-ethereum.

	cpp-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	cpp-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/**
 * @author Christian <c@ethdev.com>
 * @date 2014
 * Routines used by both the compiler and the expression compiler.
 */

#include <libsolidity/CompilerUtils.h>
#include <libsolidity/AST.h>
#include <libevmcore/Instruction.h>

using namespace std;

namespace dev
{
namespace solidity
{

const unsigned int CompilerUtils::dataStartOffset = 4;

unsigned CompilerUtils::loadFromMemory(unsigned _offset, Type const& _type,
	bool _fromCalldata, bool _padToWordBoundaries)
{
	solAssert(_type.getCategory() != Type::Category::Array, "Unable to statically load dynamic type.");
	m_context << u256(_offset);
	return loadFromMemoryHelper(_type, _fromCalldata, _padToWordBoundaries);
}

void CompilerUtils::loadFromMemoryDynamic(Type const& _type, bool _fromCalldata, bool _padToWordBoundaries)
{
	solAssert(_type.getCategory() != Type::Category::Array, "Arrays not yet implemented.");
	m_context << eth::Instruction::DUP1;
	unsigned numBytes = loadFromMemoryHelper(_type, _fromCalldata, _padToWordBoundaries);
	// update memory counter
	for (unsigned i = 0; i < _type.getSizeOnStack(); ++i)
		m_context << eth::swapInstruction(1 + i);
	m_context << u256(numBytes) << eth::Instruction::ADD;
}


unsigned CompilerUtils::storeInMemory(unsigned _offset, Type const& _type, bool _padToWordBoundaries)
{
	solAssert(_type.getCategory() != Type::Category::Array, "Unable to statically store dynamic type.");
	unsigned numBytes = prepareMemoryStore(_type, _padToWordBoundaries);
	if (numBytes > 0)
		m_context << u256(_offset) << eth::Instruction::MSTORE;
	return numBytes;
}

void CompilerUtils::storeInMemoryDynamic(Type const& _type, bool _padToWordBoundaries)
{
	if (_type.getCategory() == Type::Category::Array)
	{
		auto const& type = dynamic_cast<ArrayType const&>(_type);
		solAssert(type.isByteArray(), "Non byte arrays not yet implemented here.");

		if (type.getLocation() == ArrayType::Location::CallData)
		{
			// stack: target source_offset source_len
			m_context << eth::Instruction::DUP1 << eth::Instruction::DUP3 << eth::Instruction::DUP5
				// stack: target source_offset source_len source_len source_offset target
				<< eth::Instruction::CALLDATACOPY
				<< eth::Instruction::DUP3 << eth::Instruction::ADD
				<< eth::Instruction::SWAP2 << eth::Instruction::POP << eth::Instruction::POP;
		}
		else
		{
			solAssert(type.getLocation() == ArrayType::Location::Storage, "Memory byte arrays not yet implemented.");
			m_context << eth::Instruction::DUP1 << eth::Instruction::SLOAD;
			// stack here: memory_offset storage_offset length_bytes
			// jump to end if length is zero
			m_context << eth::Instruction::DUP1 << eth::Instruction::ISZERO;
			eth::AssemblyItem loopEnd = m_context.newTag();
			m_context.appendConditionalJumpTo(loopEnd);
			// compute memory end offset
			m_context << eth::Instruction::DUP3 << eth::Instruction::ADD << eth::Instruction::SWAP2;
			// actual array data is stored at SHA3(storage_offset)
			m_context << eth::Instruction::SWAP1;
			CompilerUtils(m_context).computeHashStatic();
			m_context << eth::Instruction::SWAP1;

			// stack here: memory_end_offset storage_data_offset memory_offset
			eth::AssemblyItem loopStart = m_context.newTag();
			m_context << loopStart
					  // load and store
					  << eth::Instruction::DUP2 << eth::Instruction::SLOAD
					  << eth::Instruction::DUP2 << eth::Instruction::MSTORE
					  // increment storage_data_offset by 1
					  << eth::Instruction::SWAP1 << u256(1) << eth::Instruction::ADD
					  // increment memory offset by 32
					  << eth::Instruction::SWAP1 << u256(32) << eth::Instruction::ADD
					  // check for loop condition
					  << eth::Instruction::DUP1 << eth::Instruction::DUP4 << eth::Instruction::GT;
			m_context.appendConditionalJumpTo(loopStart);
			m_context << loopEnd << eth::Instruction::POP << eth::Instruction::POP;
		}
	}
	else
	{
		unsigned numBytes = prepareMemoryStore(_type, _padToWordBoundaries);
		if (numBytes > 0)
		{
			solAssert(_type.getSizeOnStack() == 1, "Memory store of types with stack size != 1 not implemented.");
			m_context << eth::Instruction::DUP2 << eth::Instruction::MSTORE;
			m_context << u256(numBytes) << eth::Instruction::ADD;
		}
	}
}

void CompilerUtils::moveToStackVariable(VariableDeclaration const& _variable)
{
	unsigned const stackPosition = m_context.baseToCurrentStackOffset(m_context.getBaseStackOffsetOfVariable(_variable));
	unsigned const size = _variable.getType()->getSizeOnStack();
	// move variable starting from its top end in the stack
	if (stackPosition - size + 1 > 16)
		BOOST_THROW_EXCEPTION(CompilerError() << errinfo_sourceLocation(_variable.getLocation())
											  << errinfo_comment("Stack too deep."));
	for (unsigned i = 0; i < size; ++i)
		m_context << eth::swapInstruction(stackPosition - size + 1) << eth::Instruction::POP;
}

void CompilerUtils::copyToStackTop(unsigned _stackDepth, Type const& _type)
{
	if (_stackDepth > 16)
		BOOST_THROW_EXCEPTION(CompilerError() << errinfo_comment("Stack too deep."));
	unsigned const size = _type.getSizeOnStack();
	for (unsigned i = 0; i < size; ++i)
		m_context << eth::dupInstruction(_stackDepth);
}

void CompilerUtils::popStackElement(Type const& _type)
{
	unsigned const size = _type.getSizeOnStack();
	for (unsigned i = 0; i < size; ++i)
		m_context << eth::Instruction::POP;
}

unsigned CompilerUtils::getSizeOnStack(vector<shared_ptr<Type const>> const& _variableTypes)
{
	unsigned size = 0;
	for (shared_ptr<Type const> const& type: _variableTypes)
		size += type->getSizeOnStack();
	return size;
}

void CompilerUtils::computeHashStatic(Type const& _type, bool _padToWordBoundaries)
{
	unsigned length = storeInMemory(0, _type, _padToWordBoundaries);
	m_context << u256(length) << u256(0) << eth::Instruction::SHA3;
}

void CompilerUtils::copyByteArrayToStorage(
	ArrayType const& _targetType, ArrayType const& _sourceType) const
{
	// stack layout: [source_ref] target_ref (top)
	// need to leave target_ref on the stack at the end
	solAssert(_targetType.getLocation() == ArrayType::Location::Storage, "");
	solAssert(_targetType.isByteArray(), "Non byte arrays not yet implemented here.");
	solAssert(_sourceType.isByteArray(), "Non byte arrays not yet implemented here.");

	switch (_sourceType.getLocation())
	{
	case ArrayType::Location::CallData:
	{
		// This also assumes that after "length" we only have zeros, i.e. it cannot be used to
		// slice a byte array from calldata.

		// stack: source_offset source_len target_ref
		// fetch old length and convert to words
		m_context << eth::Instruction::DUP1 << eth::Instruction::SLOAD;
		m_context << u256(31) << eth::Instruction::ADD
				  << u256(32) << eth::Instruction::SWAP1 << eth::Instruction::DIV;
		// stack here: source_offset source_len target_ref target_length_words
		// actual array data is stored at SHA3(storage_offset)
		m_context << eth::Instruction::DUP2;
		CompilerUtils(m_context).computeHashStatic();
		// compute target_data_end
		m_context << eth::Instruction::DUP1 << eth::Instruction::SWAP2 << eth::Instruction::ADD
				  << eth::Instruction::SWAP1;
		// stack here: source_offset source_len target_ref target_data_end target_data_ref
		// store length (in bytes)
		m_context << eth::Instruction::DUP4 << eth::Instruction::DUP1 << eth::Instruction::DUP5
			<< eth::Instruction::SSTORE;
		// jump to end if length is zero
		m_context << eth::Instruction::ISZERO;
		eth::AssemblyItem copyLoopEnd = m_context.newTag();
		m_context.appendConditionalJumpTo(copyLoopEnd);
		// store start offset
		m_context << eth::Instruction::DUP5;
		// stack now: source_offset source_len target_ref target_data_end target_data_ref calldata_offset
		eth::AssemblyItem copyLoopStart = m_context.newTag();
		m_context << copyLoopStart
				  // copy from calldata and store
				  << eth::Instruction::DUP1 << eth::Instruction::CALLDATALOAD
				  << eth::Instruction::DUP3 << eth::Instruction::SSTORE
				  // increment target_data_ref by 1
				  << eth::Instruction::SWAP1 << u256(1) << eth::Instruction::ADD
				  // increment calldata_offset by 32
				  << eth::Instruction::SWAP1 << u256(32) << eth::Instruction::ADD
				  // check for loop condition
				  << eth::Instruction::DUP1 << eth::Instruction::DUP6 << eth::Instruction::GT;
		m_context.appendConditionalJumpTo(copyLoopStart);
		m_context << eth::Instruction::POP;
		m_context << copyLoopEnd;

		// now clear leftover bytes of the old value
		// stack now: source_offset source_len target_ref target_data_end target_data_ref
		clearStorageLoop();
		// stack now: source_offset source_len target_ref target_data_end

		m_context << eth::Instruction::POP << eth::Instruction::SWAP2
			<< eth::Instruction::POP << eth::Instruction::POP;
		break;
	}
	case ArrayType::Location::Storage:
	{
		// this copies source to target and also clears target if it was larger

		// stack: source_ref target_ref
		// store target_ref
		m_context << eth::Instruction::SWAP1 << eth::Instruction::DUP2;
		// fetch lengthes
		m_context << eth::Instruction::DUP1 << eth::Instruction::SLOAD << eth::Instruction::SWAP2
				  << eth::Instruction::DUP1 << eth::Instruction::SLOAD;
		// stack: target_ref target_len_bytes target_ref source_ref source_len_bytes
		// store new target length
		m_context << eth::Instruction::DUP1 << eth::Instruction::DUP4 << eth::Instruction::SSTORE;
		// compute hashes (data positions)
		m_context << eth::Instruction::SWAP2;
		CompilerUtils(m_context).computeHashStatic();
		m_context << eth::Instruction::SWAP1;
		CompilerUtils(m_context).computeHashStatic();
		// stack: target_ref target_len_bytes source_len_bytes target_data_pos source_data_pos
		// convert lengthes from bytes to storage slots
		m_context << u256(31) << u256(32) << eth::Instruction::DUP1 << eth::Instruction::DUP3
				  << eth::Instruction::DUP8 << eth::Instruction::ADD << eth::Instruction::DIV
				  << eth::Instruction::SWAP2
				  << eth::Instruction::DUP6 << eth::Instruction::ADD << eth::Instruction::DIV;
		// stack: target_ref target_len_bytes source_len_bytes target_data_pos source_data_pos target_len source_len
		// @todo we might be able to go without a third counter
		m_context << u256(0);
		// stack: target_ref target_len_bytes source_len_bytes target_data_pos source_data_pos target_len source_len counter
		eth::AssemblyItem copyLoopStart = m_context.newTag();
		m_context << copyLoopStart;
		// check for loop condition
		m_context << eth::Instruction::DUP1 << eth::Instruction::DUP3
				   << eth::Instruction::GT << eth::Instruction::ISZERO;
		eth::AssemblyItem copyLoopEnd = m_context.newTag();
		m_context.appendConditionalJumpTo(copyLoopEnd);
		// copy
		m_context << eth::Instruction::DUP4 << eth::Instruction::DUP2 << eth::Instruction::ADD
				  << eth::Instruction::SLOAD
				  << eth::Instruction::DUP6 << eth::Instruction::DUP3 << eth::Instruction::ADD
				  << eth::Instruction::SSTORE;
		// increment
		m_context << u256(1) << eth::Instruction::ADD;
		m_context.appendJumpTo(copyLoopStart);
		m_context << copyLoopEnd;

		// zero-out leftovers in target
		// stack: target_ref target_len_bytes source_len_bytes target_data_pos source_data_pos target_len source_len counter
		// add counter to target_data_pos
		m_context << eth::Instruction::DUP5 << eth::Instruction::ADD
				  << eth::Instruction::SWAP5 << eth::Instruction::POP;
		// stack: target_ref target_len_bytes target_data_pos_updated target_data_pos source_data_pos target_len source_len
		// add length to target_data_pos to get target_data_end
		m_context << eth::Instruction::POP << eth::Instruction::DUP3 << eth::Instruction::ADD
				  << eth::Instruction::SWAP4
				  << eth::Instruction::POP  << eth::Instruction::POP << eth::Instruction::POP;
		// stack: target_ref target_data_end target_data_pos_updated
		clearStorageLoop();
		m_context << eth::Instruction::POP;
		break;
	}
	default:
		solAssert(false, "Given byte array location not implemented.");
	}
}

unsigned CompilerUtils::loadFromMemoryHelper(Type const& _type, bool _fromCalldata, bool _padToWordBoundaries)
{
	unsigned _encodedSize = _type.getCalldataEncodedSize();
	unsigned numBytes = _padToWordBoundaries ? getPaddedSize(_encodedSize) : _encodedSize;
	bool leftAligned = _type.getCategory() == Type::Category::String;
	if (numBytes == 0)
		m_context << eth::Instruction::POP << u256(0);
	else
	{
		solAssert(numBytes <= 32, "Static memory load of more than 32 bytes requested.");
		m_context << (_fromCalldata ? eth::Instruction::CALLDATALOAD : eth::Instruction::MLOAD);
		if (numBytes != 32)
		{
			// add leading or trailing zeros by dividing/multiplying depending on alignment
			u256 shiftFactor = u256(1) << ((32 - numBytes) * 8);
			m_context << shiftFactor << eth::Instruction::SWAP1 << eth::Instruction::DIV;
			if (leftAligned)
				m_context << shiftFactor << eth::Instruction::MUL;
		}
	}

	return numBytes;
}

void CompilerUtils::clearByteArray(ArrayType const& _type) const
{
	solAssert(_type.getLocation() == ArrayType::Location::Storage, "");
	solAssert(_type.isByteArray(), "Non byte arrays not yet implemented here.");

	// fetch length
	m_context << eth::Instruction::DUP1 << eth::Instruction::SLOAD;
	// set length to zero
	m_context << u256(0) << eth::Instruction::DUP3 << eth::Instruction::SSTORE;
	// convert length from bytes to storage slots
	m_context << u256(31) << eth::Instruction::ADD
			  << u256(32) << eth::Instruction::SWAP1 << eth::Instruction::DIV;
	// compute data positions
	m_context << eth::Instruction::SWAP1;
	CompilerUtils(m_context).computeHashStatic();
	// stack: len data_pos
	m_context << eth::Instruction::SWAP1 << eth::Instruction::DUP2 << eth::Instruction::ADD
			  << eth::Instruction::SWAP1;
	clearStorageLoop();
	// cleanup
	m_context << eth::Instruction::POP;
}

unsigned CompilerUtils::prepareMemoryStore(Type const& _type, bool _padToWordBoundaries) const
{
	unsigned _encodedSize = _type.getCalldataEncodedSize();
	unsigned numBytes = _padToWordBoundaries ? getPaddedSize(_encodedSize) : _encodedSize;
	bool leftAligned = _type.getCategory() == Type::Category::String;
	if (numBytes == 0)
		m_context << eth::Instruction::POP;
	else
	{
		solAssert(numBytes <= 32, "Memory store of more than 32 bytes requested.");
		if (numBytes != 32 && !leftAligned && !_padToWordBoundaries)
			// shift the value accordingly before storing
			m_context << (u256(1) << ((32 - numBytes) * 8)) << eth::Instruction::MUL;
	}
	return numBytes;
}

void CompilerUtils::clearStorageLoop() const
{
	// stack: end_pos pos
	eth::AssemblyItem loopStart = m_context.newTag();
	m_context << loopStart;
	// check for loop condition
	m_context << eth::Instruction::DUP1 << eth::Instruction::DUP3
			   << eth::Instruction::GT << eth::Instruction::ISZERO;
	eth::AssemblyItem zeroLoopEnd = m_context.newTag();
	m_context.appendConditionalJumpTo(zeroLoopEnd);
	// zero out
	m_context << u256(0) << eth::Instruction::DUP2 << eth::Instruction::SSTORE;
	// increment
	m_context << u256(1) << eth::Instruction::ADD;
	m_context.appendJumpTo(loopStart);
	// cleanup
	m_context << zeroLoopEnd;
	m_context << eth::Instruction::POP;
}

}
}
