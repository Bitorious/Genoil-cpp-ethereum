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
	
class Connection: public std::enable_shared_from_this<Connection>
{
public:
	static void doAccept(bi::tcp::acceptor& _acceptor, std::function<void(std::shared_ptr<Connection>)> _success);
	
	/// Constructor for incoming connections. Socket is provided.
	Connection(boost::asio::io_service& _io_service): m_socket(_io_service) {}
	
	/// Constructor for outgoing connections.
	Connection(boost::asio::io_service& _io_service, boost::asio::ip::tcp::endpoint): m_socket(_io_service) {}

	/// Destructor.
	~Connection() { drop(); }

	boost::asio::ip::tcp::endpoint remote() { return m_socket.remote_endpoint(); }
	
protected:
	void drop();
	boost::asio::ip::tcp::socket m_socket;
};
	
/**
 * @brief Network Class
 * Network operations and interface for establishing and maintaining network connections.
 */
class Network: public Worker
{
	static constexpr unsigned c_runInterval = 10;
public:
	Network(NetworkPreferences const& _n = NetworkPreferences(), bool _start = false);
	virtual ~Network();
	
	/// Start network (blocking).
	void start() { startWorking(); };
	
	/// Stop network (blocking).
	void stop();
	
protected:
	/// Called by after network is setup but before any peer connection is established.
	virtual void onStartup() { }

	/// legacy. Called by runtime.
	virtual void onRun() {}

	/// Must be thread-safe. Called when new TCP connection is established.
	virtual void onConnection(std::shared_ptr<Connection>) {}

	/// Called by network during shutdown; returning false signals network to poll and try again. returning true signals that implementation has shutdown.
	virtual void onShutdown() {}
	
private:
	/// Runtime for network management events.
	void run(boost::system::error_code const&);
	
	virtual void startedWorking() final;		///< Called by Worker thread after start() called.
	virtual void doneWorking() final;			///< Called by Worker thread after stop() called. Shuts down network.

	NetworkPreferences m_netPrefs;			///< Network settings.
	std::unique_ptr<NetworkUtil> m_host;		///< Host addresses, upnp, etc.
	ba::io_service m_io;						///< IOService for network stuff.
	bi::tcp::acceptor m_acceptorV4;			///< IPv4 Listening acceptor.
	
	bi::tcp::endpoint m_peerAddress;
	
	Mutex x_run;											///< Prevents concurrent network start.
	bool m_run = false;									///< Indicates network is running if true, or signals network to shutdown if false.
	std::unique_ptr<boost::asio::deadline_timer> m_timer;	///< Timer for scheduling network management runtime.
};

}
}
