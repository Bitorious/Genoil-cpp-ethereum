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
 * @author Gav Wood <i@gavwood.com>
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#pragma once

#include <libdevcore/RLP.h>
#include <libdevcrypto/SHA3.h>
#include <libdevcrypto/Common.h>
#include "Network.h"

namespace dev
{
namespace p2p
{

struct OverlayPeerInfo
{
	NodeId id;
	std::string clientVersion;
	bi::tcp::endpoint ipEndpoint;
	std::chrono::steady_clock::duration lastPing;
	std::set<CapDesc> caps;
};

struct OverlayNode
{
	NodeId id;										///< Their id/public key.
	Address address;									///< Address
	bi::tcp::endpoint ipEndpoint;						///< As reported from the node itself.
	Secret token;									///< Session token.
	int score = 0;									///< All time cumulative.
	int rating = 0;									///< Trending.
	bool required;									///< If true, one or more protocols is requiring this node to be a peer.
	bool dead = false;								///< If true, we believe this node is permanently dead - forget all about it.
	std::chrono::system_clock::time_point lastConnected;
	std::chrono::system_clock::time_point lastAttempted;
	unsigned failedAttempts = 0;
	DisconnectReason lastDisconnect = NoDisconnect;	///< Reason for disconnect that happened last.

	Origin idOrigin = Origin::Unknown;				///< How did we get to know this node's id?

	int secondsSinceLastConnected() const { return lastConnected == std::chrono::system_clock::time_point() ? -1 : (int)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - lastConnected).count(); }
	int secondsSinceLastAttempted() const { return lastAttempted == std::chrono::system_clock::time_point() ? -1 : (int)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - lastAttempted).count(); }

	unsigned fallbackSeconds() const;
	bool shouldReconnect() const;
	
	bool isOffline() const { return !token || lastAttempted > lastConnected; }
	virtual bool operator<(OverlayNode const& _n) const
	{
		if (isOffline() != _n.isOffline())
			return isOffline();
		else if (isOffline())
			if (lastAttempted == _n.lastAttempted)
				return failedAttempts < _n.failedAttempts;
			else
				return lastAttempted < _n.lastAttempted;
		else
			if (score == _n.score)
				if (rating == _n.rating)
					return failedAttempts < _n.failedAttempts;
				else
					return rating < _n.rating;
			else
				return score < _n.score;
	}
};

class PeerProtocol;

/**
 * @brief Peer of Overlay Network
 * Implementation of Network to provide P2P Network Overlay
 * @todo constructor: restoreNodes, capabilities
 * @todo mutex: capabilities, nodes, ratings (peer = node+cap+rating)
 */
class PeerOverlay: public Network
{
	static constexpr unsigned protocolVersion = 3;
	static constexpr unsigned kBuckets = 20;
public:
	/// Start peer network, listening for connections if start is true.
	PeerOverlay(std::string const& _clientVersion, NetworkPreferences const& _n = NetworkPreferences(), bool _start = false);
	
	NodeId id() const { return m_alias.pub(); }
	
	/// Register a peer-capability; all new peer connections will have this capability.
	template <class T> std::weak_ptr<T> registerCapability(T* _t) { auto ret = std::shared_ptr<T>(_t); m_capabilities[T::capDesc()] = ret; return ret; }
	
	/// Used during handshake to determine if capability is supported.
	bool haveCapability(CapDesc const& _name) const { return m_capabilities.count(_name) != 0; }
	
	/// Returns peers of capability.
	template <class T> PeerInfos peers() const { return (*m_peers)[T::capDesc()]; }
	
	/// @returns true iff we have a peer of the given id.
	template <class T> bool havePeer(NodeId _id) const { return (*m_peers)[T::capDesc()].count(_id); }

	template <class T> void send(NodeId, RLPStream);
	template <class T> void makePeer(NodeId);
	template <class T> void notRequired(NodeId);
	
protected:
	virtual void onStartup();
	virtual void onRun();
	virtual void onConnect(std::shared_ptr<Connection>);
	virtual void onShutdown();
	
	u160 dist(NodeId const& _n) const { return right160(sha3(id()))^right160(sha3(_n)); }
	
	unsigned int binDist(NodeId const& _n) const { auto d = dist(_n); unsigned ret; for (ret = 0; d >>= 1; ++ret) {}; return ret; }
	
	/// If leastSeen doesn't respond then kick it from m_table, otherwise leastSeen will be removed.
	void keepOrKill(NodeId _new, NodeId _leastSeen);
	
	void noteNode(NodeId _n) { auto &t = m_state[binDist(_n)]; t.remove(_n); if (t.size() < kBuckets - 1) t.push_back(_n); else keepOrKill(_n, t.front()); }

private:
	std::string m_clientVersion;		///< Our version string.
	KeyPair m_alias;					///< Our default network alias.
	
	std::map<CapDesc, std::shared_ptr<PeerProtocol>> m_capabilities;	///< Each of the capabilities we support.
	
	std::map<unsigned, std::list<NodeId>> m_state;	///< State of this node (kademlia xor metric).
	
	mutable Mutex x_nodes;
	std::map<NodeId, std::pair<OverlayNode, std::weak_ptr<Connection>>> m_nodes;
	
	/// Status of nodes to which we are currently connected.
	std::unique_ptr<std::map<CapDesc, std::vector<PeerInfo>>> m_peers;
};

}
}
