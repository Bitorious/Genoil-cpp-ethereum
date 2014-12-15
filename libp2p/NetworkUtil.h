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
/** @file NetworkUtil.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#pragma once

#include "Common.h"
namespace ba = boost::asio;
namespace bi = ba::ip;


namespace dev
{
namespace p2p
{

/**
 * @brief Abstraction of static host network & interfaces (TCP/IP).
 * @todo UDP, ICMP
 * @todo ifup/ifdown events
 */
struct NetworkUtil
{
	/// @returns public and private interface addresses
	static std::vector<bi::address> interfaceAddresses();
	
	/// Return public endpoint of upnp interface. If successful o_upnpifaddr will be a private interface address and endpoint will contain public address and port.
	static bi::tcp::endpoint traverseNAT(std::vector<bi::address> const& _ifAddresses, unsigned short _listenPort);
	
	NetworkUtil(): ifAddresses(interfaceAddresses()) {}
	
	/// @returns *public* endpoint and updates potential publicAddresses. Attempts binding to _prefs.listenPort, else attempt via net-allocated port. Not thread-safe.
	/// Endpoint precedence: User Provided > Public > UPnP [> Private] > Unspecified
	bi::tcp::endpoint listen4(NetworkPreferences const& _prefs, bi::tcp::acceptor& _acceptor);
	
	std::vector<bi::address> ifAddresses;		///< Interface addresses (private, public).
	std::set<bi::address> publicAddresses;		///< Public addresses that peers (can) know us by.
};
	
}
}
