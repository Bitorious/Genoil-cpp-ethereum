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
/** @file Network.h
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#pragma once

#include <libdevcore/Worker.h>
#include "NetworkUtil.h"
#include "Common.h"
namespace ba = boost::asio;
namespace bi = ba::ip;

namespace dev
{
namespace p2p
{
	
class Connection;

/**
 * @brief Network Class
 * Network operations and interface for establishing and maintaining network connections.
 * @todo udp4/6, tcp6
 * @todo Abstract interface for endpoints (acceptors and addresses for ipv4, ipv6, tcp, and udp).
 * @todo ringbuffer or linkedlist for network-event ipc
 */
class Network: virtual public Worker
{
	static constexpr unsigned c_maintenanceInterval = 10;
	friend class Connection;
public:
	Network(NetworkPreferences const& _n = NetworkPreferences(), bool _start = false);
	virtual ~Network() {};
	
	/// Start network (blocking).
	void start() { startWorking(); };
	
	/// Stop network (blocking).
	void stop();

	/// Accept incoming connections.
	void doAccept(bi::tcp::acceptor& _acceptor);
	
	/// Connect to remote endpoint.
	void connect(boost::asio::ip::tcp::endpoint _remote);
	
protected:
	/// Endpoint which we are accepting connections on.
	bi::tcp::endpoint tcp4Endpoint() { return m_tcp4Endpoint; }
	
	/// Called after network is setup. (todo: before connections are created or accepted)
	virtual void onStartup() { }

	/// legacy. Called by runtime at c_maintenanceInterval + time spent running.
	virtual void onRun() {}

	/// Called by runtime when new TCP connection is established.
	virtual void onConnect(std::shared_ptr<Connection>) {}

	/// Called by network during shutdown; returning false signals network to poll and try again. returning true signals that implementation has shutdown.
	virtual void onShutdown() {}
	
private:
	/// Runtime for network events. (todo: runs *only* when event is pending, similar to connection write queueing)
	void run(boost::system::error_code const&);
	
	virtual void startedWorking() final;		///< Called by Worker thread after start() called.
	virtual void doneWorking() final;			///< Called by Worker thread after stop() called. Shuts down network.

	NetworkPreferences m_netPrefs;			///< Network settings.
	std::unique_ptr<NetworkUtil> m_host;		///< Host addresses, upnp, etc.
	ba::io_service m_io;						///< IOService for network stuff.
	bi::tcp::acceptor m_tcp4Acceptor;			///< IPv4 Listening acceptor.
	bi::tcp::endpoint m_tcp4Endpoint;			///< IPv4 Address we advertise for ingress connections.
	
	Mutex x_run;											///< Prevents concurrent network start.
	bool m_run = false;									///< Indicates network is running if true, or signals network to shutdown if false.
	std::unique_ptr<boost::asio::deadline_timer> m_timer;	///< Timer for scheduling network management runtime.
};

class Connection: public std::enable_shared_from_this<Connection>
{
	friend class Network;
public:
	/// Constructor for incoming connections.
	Connection(Network& _network): m_originated(false), m_socket(_network.m_io) {}
	
	/// Constructor for outgoing connections.
	Connection(Network& _network, boost::asio::ip::tcp::endpoint): m_originated(true), m_socket(_network.m_io) {}

	/// Destructor.
	~Connection() { close(); }

	boost::asio::ip::tcp::endpoint remote() { return m_socket.remote_endpoint(); }
	
protected:
	void close();
	
private:
	bool m_originated;
	boost::asio::ip::tcp::socket m_socket;
};
	
class RLPXConnection: public Connection
{
	// implements handshake when connected
};
	
}
}
