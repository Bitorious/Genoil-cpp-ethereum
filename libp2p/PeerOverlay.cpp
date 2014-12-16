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
/** @file PeerOverlay.cpp
 * @author Gav Wood <i@gavwood.com>
 * @author Alex Leverington <nessence@gmail.com>
 * @date 2014
 */

#include <libdevcore/Guards.h>
#include "PeerOverlay.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;

PeerOverlay::PeerOverlay(string const& _clientVersion, NetworkPreferences const& _n, bool _start):
	Worker("p2p", 0),
	Network(_n, _start),
	m_clientVersion(_clientVersion)
{
	for (auto i = 0; i < Address::size; i++)
		m_state[i] = std::list<NodeId>();
}

void PeerOverlay::onStartup()
{
// Overlay:
//	// start capability threads
//	for (auto const& h: m_capabilities)
//		h.second->onStarting();
//	// if m_public address is valid then add us to node list
//	// todo: abstract empty() and emplace logic
//	if (!m_public.address().is_unspecified() && (m_nodes.empty() || m_nodes[m_nodesList[0]]->id != id()))
//		noteNode(id(), m_public, Origin::Perfect, false);
//	clog(NetNote) << "Id:" << id().abridged();
}

void PeerOverlay::onRun()
{
// Overlay:
//	event: grow/prune nodes
//	event: ping peers
}

void PeerOverlay::onConnect(std::shared_ptr<Connection>)
{
// Overlay:
//	auto p = std::make_shared<Session>(this, std::move(*m_socket.release()), bi::tcp::endpoint(remoteAddress, 0));
//	p->start();
}

void PeerOverlay::onShutdown()
{
// Overlay:
//	// stop capabilities (eth: stops syncing or block/tx broadcast)
//	for (auto const& h: m_capabilities)
//		h.second->onStopping();
//
//	// disconnect peers
//	for (unsigned n = 0;; n = 0)
//	{
//		{
//			RecursiveGuard l(x_peers);
//			for (auto i: m_peers)
//				if (auto p = i.second.lock())
//					if (p->isOpen())
//					{
//						p->disconnect(ClientQuit);
//						n++;
//					}
//		}
//		if (!n)
//			break;
//
//		// poll until peers send out disconnect packets and are dropped
//		m_io.poll();
//	}
}
