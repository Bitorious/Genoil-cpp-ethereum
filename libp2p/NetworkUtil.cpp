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
/** @file NetworkUtil.cpp
 * @author Alex Leverington <nessence@gmail.com>
 * @author Gav Wood <i@gavwood.com>
 * @author Eric Lombrozo <elombrozo@gmail.com> (Windows version of interfaceAddresses())
 * @date 2014
 */

#include <sys/types.h>
#ifndef _WIN32
#include <ifaddrs.h>
#endif

#include <boost/algorithm/string.hpp>
#include <libethcore/Exceptions.h>
#include "UPnP.h"
#include "Network.h"

using namespace std;
using namespace dev;
using namespace dev::p2p;

std::vector<bi::address> NetworkUtil::interfaceAddresses()
{
	std::vector<bi::address> addresses;

#ifdef _WIN32
	WSAData wsaData;
	if (WSAStartup(MAKEWORD(1, 1), &wsaData) != 0)
		BOOST_THROW_EXCEPTION(NoNetworking());
	
	char ac[80];
	if (gethostname(ac, sizeof(ac)) == SOCKET_ERROR)
	{
		clog(NetWarn) << "Error " << WSAGetLastError() << " when getting local host name.";
		WSACleanup();
		BOOST_THROW_EXCEPTION(NoNetworking());
	}
	
	struct hostent* phe = gethostbyname(ac);
	if (phe == 0)
	{
		clog(NetWarn) << "Bad host lookup.";
		WSACleanup();
		BOOST_THROW_EXCEPTION(NoNetworking());
	}
	
	for (int i = 0; phe->h_addr_list[i] != 0; ++i)
	{
		struct in_addr addr;
		memcpy(&addr, phe->h_addr_list[i], sizeof(struct in_addr));
		char *addrStr = inet_ntoa(addr);
		bi::address address(bi::address::from_string(addrStr));
		if (!isLocalHostAddress(address))
			addresses.push_back(address.to_v4());
	}
	
	WSACleanup();
#else
	ifaddrs* ifaddr;
	if (getifaddrs(&ifaddr) == -1)
		BOOST_THROW_EXCEPTION(NoNetworking());
	
	for (auto ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next)
	{
		if (!ifa->ifa_addr || string(ifa->ifa_name) == "lo0")
			continue;
		
		if (ifa->ifa_addr->sa_family == AF_INET)
		{
			in_addr addr = ((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
			boost::asio::ip::address_v4 address(boost::asio::detail::socket_ops::network_to_host_long(addr.s_addr));
			if (!isLocalHostAddress(address))
				addresses.push_back(address);
		}
		else if (ifa->ifa_addr->sa_family == AF_INET6)
		{
			sockaddr_in6* sockaddr = ((struct sockaddr_in6 *)ifa->ifa_addr);
			in6_addr addr = sockaddr->sin6_addr;
			boost::asio::ip::address_v6::bytes_type bytes;
			memcpy(&bytes[0], addr.s6_addr, 16);
			boost::asio::ip::address_v6 address(bytes, sockaddr->sin6_scope_id);
			if (!isLocalHostAddress(address))
				addresses.push_back(address);
		}
	}
	
	if (ifaddr!=NULL)
		freeifaddrs(ifaddr);
	
#endif
	
	return std::move(addresses);
}

bi::tcp::endpoint NetworkUtil::traverseNAT(std::vector<bi::address> const& _ifAddresses, unsigned short _listenPort)
{
	asserts(_listenPort != 0);
	
	UPnP* upnp = nullptr;
	try
	{
		upnp = new UPnP;
	}
	// let m_upnp continue as null - we handle it properly.
	catch (NoUPnPDevice) {}
	
	bi::tcp::endpoint upnpep;
	if (upnp && upnp->isValid())
	{
		bi::address paddr;
		int extPort = 0;
		for (auto const& addr: _ifAddresses)
			if (addr.is_v4() && isPrivateAddress(addr) && (extPort = upnp->addRedirect(addr.to_string().c_str(), _listenPort)))
			{
				paddr = addr;
				break;
			}
		
		auto eip = upnp->externalIP();
		bi::address eipaddr(bi::address::from_string(eip));
		if (extPort && eip != string("0.0.0.0") && !isPrivateAddress(eipaddr))
		{
			clog(NetNote) << "Punched through NAT and mapped local port" << _listenPort << "onto external port" << extPort << ".";
			clog(NetNote) << "External addr:" << eip;
			// todo: [test] Is extip invalid when local ifaddr returned by upnp is unspecified?
			upnpep = bi::tcp::endpoint(eipaddr, (unsigned short)extPort);
		}
		else
			clog(NetWarn) << "Couldn't punch through NAT (or no NAT in place).";
		
		if (upnp)
			delete upnp;
	}
	
	return upnpep;
}

bi::tcp::endpoint NetworkUtil::listen4(NetworkPreferences const& _prefs, bi::tcp::acceptor& _acceptor)
{
	if (!ifAddresses.size())
		return bi::tcp::endpoint();
	
	int retport = -1;
	for (unsigned i = 0; i < 2; ++i)
	{
		// try to connect w/listenPort, else attempt net-allocated port
		bi::tcp::endpoint endpoint(bi::tcp::v4(), i ? 0 : _prefs.listenPort);
		try
		{
			_acceptor.open(endpoint.protocol());
			_acceptor.set_option(ba::socket_base::reuse_address(true));
			_acceptor.bind(endpoint);
			_acceptor.listen();
			retport = _acceptor.local_endpoint().port();
			break;
		}
		catch (...)
		{
			if (i)
				// both attempts failed
				cwarn << "Couldn't start accepting connections on host. Something very wrong with network?\n" << boost::current_exception_diagnostic_information();
			
			// first attempt failed
			_acceptor.close();
		}
	}
	
	// no point continuing if there are no interface addresses or valid listen port
	if (retport < 1)
		return bi::tcp::endpoint();
	
	// populate interfaces _acceptor listens on; excluding local if applicable
	for (auto addr: ifAddresses)
		if ((_prefs.localNetworking || !isPrivateAddress(addr)) && !isLocalHostAddress(addr))
			publicAddresses.insert(addr);
	
	// if user supplied address is a public address then we use it
	// if user supplied address is private, and localnetworking is enabled, we use it
	bi::address reqpublicaddr(bi::address(_prefs.publicIP.empty() ? bi::address() : bi::address::from_string(_prefs.publicIP)));
	bi::tcp::endpoint reqpublic(reqpublicaddr, retport);
	bool isprivate = isPrivateAddress(reqpublicaddr);
	bool ispublic = !isprivate && !isLocalHostAddress(reqpublicaddr);
	if (!reqpublicaddr.is_unspecified() && (ispublic || (isprivate && _prefs.localNetworking)))
	{
		if (!publicAddresses.count(reqpublicaddr))
			publicAddresses.insert(reqpublicaddr);
		return reqpublic;
	}
	
	// if address wasn't provided, then use first public ipv4 address found
	for (auto addr: publicAddresses)
		if (addr.is_v4() && !isPrivateAddress(addr))
			return bi::tcp::endpoint(*publicAddresses.begin(), retport);
	
	// or find address via upnp
	if (_prefs.upnp)
	{
		bi::address upnpifaddr;
		bi::tcp::endpoint upnpep = traverseNAT(ifAddresses, retport);
		if (!upnpep.address().is_unspecified())
		{
			if (!publicAddresses.count(upnpep.address()))
				publicAddresses.insert(upnpep.address());
			return upnpep;
		}
	}
	
	// or if no address provided, use private ipv4 address if local networking is enabled
	if (reqpublicaddr.is_unspecified() && _prefs.localNetworking)
		for (auto addr: publicAddresses)
			if (addr.is_v4() && isPrivateAddress(addr))
				return bi::tcp::endpoint(addr, retport);
	
	// otherwise address is unspecified
	return bi::tcp::endpoint(bi::address(), retport);
}

