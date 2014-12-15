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
/** @file PeerOverlay.h
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#pragma once

#include "Network.h"

namespace dev
{
namespace p2p
{

class PeerProtocol;
	
/**
 * @brief Peer of Overlay Network
 * Implementation of Network to provide P2P Network Overlay
 * @todo constructor: restoreNodes, capabilities
 * @todo mutex: capabilities, nodes, ratings (peer = node+cap+rating)
 */
class PeerOverlay: Network
{
	/// Start peer network, listening for connections if start is true.
	PeerOverlay(std::string const& _clientVersion, NetworkPreferences const& _n = NetworkPreferences(), bool _start = false);
	
	static constexpr unsigned protocolVersion = 3;
	
	/// Register a peer-capability; all new peer connections will have this capability.
	template <class T> std::weak_ptr<T> registerCapability(T* _t) { auto ret = std::shared_ptr<T>(_t); m_capabilities[std::make_pair(T::staticName(), T::staticVersion())] = ret; return ret; }
	
	bool haveCapability(CapDesc const& _name) const { return m_capabilities.count(_name) != 0; }
	
protected:
	virtual void onStartup();
	virtual void onRun();
	virtual void onConnection(std::shared_ptr<Connection>);
	virtual void onShutdown();
	
//	/// Get peer information.
//	PeerInfos peers(bool _updatePing = false) const;
//	/// Serialise the set of known peers.
//	bytes saveNodes() const;
//	
//	/// Deserialise the data and populate the set of known peers.
//	void restoreNodes(bytesConstRef _b);

//	Nodes nodes() const { RecursiveGuard l(x_peers); Nodes ret; for (auto const& i: m_nodes) ret.push_back(*i.second); return ret; }
	
private:
//	NodeId id() const { return m_key.pub(); }
//	virtual void doWork();
//	void ensurePeers();
//	/// @returns true iff we have a peer of the given id.
//	bool havePeer(NodeId _id) const;

	std::map<CapDesc, std::shared_ptr<PeerProtocol>> m_capabilities;	///< Each of the capabilities we support.
	std::map<CapDesc, unsigned> m_capabilitiesIdealPeerCount;			///< Ideal peer count for each capability.
};

}
}
