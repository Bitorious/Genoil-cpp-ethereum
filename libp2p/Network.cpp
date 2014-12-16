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
/** @file Network.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @date 2014
 */

#include <libethcore/Exceptions.h>
#include "Network.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;

Network::Network(NetworkPreferences const& _n, bool _start):
	Worker("net", 0),
	m_netPrefs(_n),
	m_io(),
	m_tcp4Acceptor(m_io)
{
	// todo: Is this safe?
	if (_start)
		start();
}

void Network::stop()
{
	{
		// when m_run == false, run() will cause this::run() to stop() ioservice
		Guard l(x_run);
		if (!m_run)
			return;
		m_run = false;
	}
	
	// wait for network scheduler to stop
	while (!!m_timer)
		this_thread::sleep_for(chrono::milliseconds(50));
	
	// stop worker thread
	stopWorking();
}

void Network::doAccept(bi::tcp::acceptor& _acceptor)
{
	auto newConn = make_shared<Connection>(*this);
	_acceptor.async_accept(newConn->m_socket, [=, &_acceptor](boost::system::error_code _ec)
	{
		if (_ec)
			newConn->close();
		else
			onConnect(newConn); // see onConnect

		if (_ec.value() < 1)
			doAccept(_acceptor);
	});
}

void Network::run(boost::system::error_code const&)
{
	if (!m_run)
	{
		m_io.stop();		// pause network I/O
		m_timer.reset();	// kill timer
		// event: cleanup (PeerWork, etc.)
		return;
	}
	
	/// todo: resize queue: ingress -> protocols
	
	onRun();
	
	/// todo: resize queue: egress <- protocols
	
	auto runcb = [this](boost::system::error_code const& error) -> void { run(error); };
	m_timer->expires_from_now(boost::posix_time::milliseconds(c_maintenanceInterval));
	m_timer->async_wait(runcb);
}

void Network::startedWorking()
{
	asserts(!m_timer);
	
	{
		// prevent m_run from being set to true at same time as set to false by stop()
		// don't release mutex until m_timer is set in case stop() is called at same
		// time, stop will wait on m_timer for graceful network shutdown.
		Guard l(x_run);
		m_timer.reset(new boost::asio::deadline_timer(m_io));
		m_run = true;
	}
	
	// try to open acceptor
	m_tcp4Endpoint = m_host->listen4(m_netPrefs, m_tcp4Acceptor);

	onStartup();
	
	// listen to connections
	// todo: GUI when listen is unavailable in UI
	if (!m_tcp4Endpoint.address().is_unspecified())
		doAccept(m_tcp4Acceptor);
	
	run(boost::system::error_code());
}

void Network::doneWorking()
{
	// Note: m_io must be reset whenever poll() stops from running out of work.

	m_io.reset();
	if (m_tcp4Acceptor.is_open())
	{
		m_tcp4Acceptor.cancel();
		m_tcp4Acceptor.close();
	}
	m_io.poll(); // poll for incoming connection which have been accepted but not started.
	
	m_io.reset(); // see note
	onShutdown();
	m_io.poll(); // poll for potential quit packets

//	m_io.stop(); // stop network (call before subsequent reset() - todo: is this necessary?)
	m_io.reset(); // for future start()
}

void Connection::close()
{
	if (m_socket.is_open())
	{
		boost::system::error_code ec;
		m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
		m_socket.close();
	}
}
