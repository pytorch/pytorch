#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for psutil.net_connections() and Process.net_connections() APIs."""

import os
import socket
import textwrap
from contextlib import closing
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_DGRAM
from socket import SOCK_STREAM

import psutil
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil.tests import AF_UNIX
from psutil.tests import HAS_NET_CONNECTIONS_UNIX
from psutil.tests import SKIP_SYSCONS
from psutil.tests import PsutilTestCase
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import filter_proc_net_connections
from psutil.tests import pytest
from psutil.tests import reap_children
from psutil.tests import retry_on_failure
from psutil.tests import skip_on_access_denied
from psutil.tests import tcp_socketpair
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file


SOCK_SEQPACKET = getattr(socket, "SOCK_SEQPACKET", object())


def this_proc_net_connections(kind):
    cons = psutil.Process().net_connections(kind=kind)
    if kind in ("all", "unix"):
        return filter_proc_net_connections(cons)
    return cons


@pytest.mark.xdist_group(name="serial")
class ConnectionTestCase(PsutilTestCase):
    def setUp(self):
        assert this_proc_net_connections(kind='all') == []

    def tearDown(self):
        # Make sure we closed all resources.
        assert this_proc_net_connections(kind='all') == []

    def compare_procsys_connections(self, pid, proc_cons, kind='all'):
        """Given a process PID and its list of connections compare
        those against system-wide connections retrieved via
        psutil.net_connections.
        """
        try:
            sys_cons = psutil.net_connections(kind=kind)
        except psutil.AccessDenied:
            # On MACOS, system-wide connections are retrieved by iterating
            # over all processes
            if MACOS:
                return
            else:
                raise
        # Filter for this proc PID and exlucde PIDs from the tuple.
        sys_cons = [c[:-1] for c in sys_cons if c.pid == pid]
        sys_cons.sort()
        proc_cons.sort()
        assert proc_cons == sys_cons


class TestBasicOperations(ConnectionTestCase):
    @pytest.mark.skipif(SKIP_SYSCONS, reason="requires root")
    def test_system(self):
        with create_sockets():
            for conn in psutil.net_connections(kind='all'):
                check_connection_ntuple(conn)

    def test_process(self):
        with create_sockets():
            for conn in this_proc_net_connections(kind='all'):
                check_connection_ntuple(conn)

    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            this_proc_net_connections(kind='???')
        with pytest.raises(ValueError):
            psutil.net_connections(kind='???')


@pytest.mark.xdist_group(name="serial")
class TestUnconnectedSockets(ConnectionTestCase):
    """Tests sockets which are open but not connected to anything."""

    def get_conn_from_sock(self, sock):
        cons = this_proc_net_connections(kind='all')
        smap = dict([(c.fd, c) for c in cons])
        if NETBSD or FREEBSD:
            # NetBSD opens a UNIX socket to /var/log/run
            # so there may be more connections.
            return smap[sock.fileno()]
        else:
            assert len(cons) == 1
            if cons[0].fd != -1:
                assert smap[sock.fileno()].fd == sock.fileno()
            return cons[0]

    def check_socket(self, sock):
        """Given a socket, makes sure it matches the one obtained
        via psutil. It assumes this process created one connection
        only (the one supposed to be checked).
        """
        conn = self.get_conn_from_sock(sock)
        check_connection_ntuple(conn)

        # fd, family, type
        if conn.fd != -1:
            assert conn.fd == sock.fileno()
        assert conn.family == sock.family
        # see: http://bugs.python.org/issue30204
        assert conn.type == sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)

        # local address
        laddr = sock.getsockname()
        if not laddr and PY3 and isinstance(laddr, bytes):
            # See: http://bugs.python.org/issue30205
            laddr = laddr.decode()
        if sock.family == AF_INET6:
            laddr = laddr[:2]
        assert conn.laddr == laddr

        # XXX Solaris can't retrieve system-wide UNIX sockets
        if sock.family == AF_UNIX and HAS_NET_CONNECTIONS_UNIX:
            cons = this_proc_net_connections(kind='all')
            self.compare_procsys_connections(os.getpid(), cons, kind='all')
        return conn

    def test_tcp_v4(self):
        addr = ("127.0.0.1", 0)
        with closing(bind_socket(AF_INET, SOCK_STREAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            assert conn.raddr == ()
            assert conn.status == psutil.CONN_LISTEN

    @pytest.mark.skipif(not supports_ipv6(), reason="IPv6 not supported")
    def test_tcp_v6(self):
        addr = ("::1", 0)
        with closing(bind_socket(AF_INET6, SOCK_STREAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            assert conn.raddr == ()
            assert conn.status == psutil.CONN_LISTEN

    def test_udp_v4(self):
        addr = ("127.0.0.1", 0)
        with closing(bind_socket(AF_INET, SOCK_DGRAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            assert conn.raddr == ()
            assert conn.status == psutil.CONN_NONE

    @pytest.mark.skipif(not supports_ipv6(), reason="IPv6 not supported")
    def test_udp_v6(self):
        addr = ("::1", 0)
        with closing(bind_socket(AF_INET6, SOCK_DGRAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            assert conn.raddr == ()
            assert conn.status == psutil.CONN_NONE

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_unix_tcp(self):
        testfn = self.get_testfn()
        with closing(bind_unix_socket(testfn, type=SOCK_STREAM)) as sock:
            conn = self.check_socket(sock)
            assert conn.raddr == ""  # noqa
            assert conn.status == psutil.CONN_NONE

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_unix_udp(self):
        testfn = self.get_testfn()
        with closing(bind_unix_socket(testfn, type=SOCK_STREAM)) as sock:
            conn = self.check_socket(sock)
            assert conn.raddr == ""  # noqa
            assert conn.status == psutil.CONN_NONE


@pytest.mark.xdist_group(name="serial")
class TestConnectedSocket(ConnectionTestCase):
    """Test socket pairs which are actually connected to
    each other.
    """

    # On SunOS, even after we close() it, the server socket stays around
    # in TIME_WAIT state.
    @pytest.mark.skipif(SUNOS, reason="unreliable on SUONS")
    def test_tcp(self):
        addr = ("127.0.0.1", 0)
        assert this_proc_net_connections(kind='tcp4') == []
        server, client = tcp_socketpair(AF_INET, addr=addr)
        try:
            cons = this_proc_net_connections(kind='tcp4')
            assert len(cons) == 2
            assert cons[0].status == psutil.CONN_ESTABLISHED
            assert cons[1].status == psutil.CONN_ESTABLISHED
            # May not be fast enough to change state so it stays
            # commenteed.
            # client.close()
            # cons = this_proc_net_connections(kind='all')
            # self.assertEqual(len(cons), 1)
            # self.assertEqual(cons[0].status, psutil.CONN_CLOSE_WAIT)
        finally:
            server.close()
            client.close()

    @pytest.mark.skipif(not POSIX, reason="POSIX only")
    def test_unix(self):
        testfn = self.get_testfn()
        server, client = unix_socketpair(testfn)
        try:
            cons = this_proc_net_connections(kind='unix')
            assert not (cons[0].laddr and cons[0].raddr), cons
            assert not (cons[1].laddr and cons[1].raddr), cons
            if NETBSD or FREEBSD:
                # On NetBSD creating a UNIX socket will cause
                # a UNIX connection to  /var/run/log.
                cons = [c for c in cons if c.raddr != '/var/run/log']
            assert len(cons) == 2
            if LINUX or FREEBSD or SUNOS or OPENBSD:
                # remote path is never set
                assert cons[0].raddr == ""  # noqa
                assert cons[1].raddr == ""  # noqa
                # one local address should though
                assert testfn == (cons[0].laddr or cons[1].laddr)
            else:
                # On other systems either the laddr or raddr
                # of both peers are set.
                assert (cons[0].laddr or cons[1].laddr) == testfn
        finally:
            server.close()
            client.close()


class TestFilters(ConnectionTestCase):
    def test_filters(self):
        def check(kind, families, types):
            for conn in this_proc_net_connections(kind=kind):
                assert conn.family in families
                assert conn.type in types
            if not SKIP_SYSCONS:
                for conn in psutil.net_connections(kind=kind):
                    assert conn.family in families
                    assert conn.type in types

        with create_sockets():
            check(
                'all',
                [AF_INET, AF_INET6, AF_UNIX],
                [SOCK_STREAM, SOCK_DGRAM, SOCK_SEQPACKET],
            )
            check('inet', [AF_INET, AF_INET6], [SOCK_STREAM, SOCK_DGRAM])
            check('inet4', [AF_INET], [SOCK_STREAM, SOCK_DGRAM])
            check('tcp', [AF_INET, AF_INET6], [SOCK_STREAM])
            check('tcp4', [AF_INET], [SOCK_STREAM])
            check('tcp6', [AF_INET6], [SOCK_STREAM])
            check('udp', [AF_INET, AF_INET6], [SOCK_DGRAM])
            check('udp4', [AF_INET], [SOCK_DGRAM])
            check('udp6', [AF_INET6], [SOCK_DGRAM])
            if HAS_NET_CONNECTIONS_UNIX:
                check(
                    'unix',
                    [AF_UNIX],
                    [SOCK_STREAM, SOCK_DGRAM, SOCK_SEQPACKET],
                )

    @skip_on_access_denied(only_if=MACOS)
    def test_combos(self):
        reap_children()

        def check_conn(proc, conn, family, type, laddr, raddr, status, kinds):
            all_kinds = (
                "all",
                "inet",
                "inet4",
                "inet6",
                "tcp",
                "tcp4",
                "tcp6",
                "udp",
                "udp4",
                "udp6",
            )
            check_connection_ntuple(conn)
            assert conn.family == family
            assert conn.type == type
            assert conn.laddr == laddr
            assert conn.raddr == raddr
            assert conn.status == status
            for kind in all_kinds:
                cons = proc.net_connections(kind=kind)
                if kind in kinds:
                    assert cons != []
                else:
                    assert cons == []
            # compare against system-wide connections
            # XXX Solaris can't retrieve system-wide UNIX
            # sockets.
            if HAS_NET_CONNECTIONS_UNIX:
                self.compare_procsys_connections(proc.pid, [conn])

        tcp_template = textwrap.dedent("""
            import socket, time
            s = socket.socket({family}, socket.SOCK_STREAM)
            s.bind(('{addr}', 0))
            s.listen(5)
            with open('{testfn}', 'w') as f:
                f.write(str(s.getsockname()[:2]))
            [time.sleep(0.1) for x in range(100)]
            """)

        udp_template = textwrap.dedent("""
            import socket, time
            s = socket.socket({family}, socket.SOCK_DGRAM)
            s.bind(('{addr}', 0))
            with open('{testfn}', 'w') as f:
                f.write(str(s.getsockname()[:2]))
            [time.sleep(0.1) for x in range(100)]
            """)

        # must be relative on Windows
        testfile = os.path.basename(self.get_testfn(dir=os.getcwd()))
        tcp4_template = tcp_template.format(
            family=int(AF_INET), addr="127.0.0.1", testfn=testfile
        )
        udp4_template = udp_template.format(
            family=int(AF_INET), addr="127.0.0.1", testfn=testfile
        )
        tcp6_template = tcp_template.format(
            family=int(AF_INET6), addr="::1", testfn=testfile
        )
        udp6_template = udp_template.format(
            family=int(AF_INET6), addr="::1", testfn=testfile
        )

        # launch various subprocess instantiating a socket of various
        # families and types to enrich psutil results
        tcp4_proc = self.pyrun(tcp4_template)
        tcp4_addr = eval(wait_for_file(testfile, delete=True))  # noqa
        udp4_proc = self.pyrun(udp4_template)
        udp4_addr = eval(wait_for_file(testfile, delete=True))  # noqa
        if supports_ipv6():
            tcp6_proc = self.pyrun(tcp6_template)
            tcp6_addr = eval(wait_for_file(testfile, delete=True))  # noqa
            udp6_proc = self.pyrun(udp6_template)
            udp6_addr = eval(wait_for_file(testfile, delete=True))  # noqa
        else:
            tcp6_proc = None
            udp6_proc = None
            tcp6_addr = None
            udp6_addr = None

        for p in psutil.Process().children():
            cons = p.net_connections()
            assert len(cons) == 1
            for conn in cons:
                # TCP v4
                if p.pid == tcp4_proc.pid:
                    check_conn(
                        p,
                        conn,
                        AF_INET,
                        SOCK_STREAM,
                        tcp4_addr,
                        (),
                        psutil.CONN_LISTEN,
                        ("all", "inet", "inet4", "tcp", "tcp4"),
                    )
                # UDP v4
                elif p.pid == udp4_proc.pid:
                    check_conn(
                        p,
                        conn,
                        AF_INET,
                        SOCK_DGRAM,
                        udp4_addr,
                        (),
                        psutil.CONN_NONE,
                        ("all", "inet", "inet4", "udp", "udp4"),
                    )
                # TCP v6
                elif p.pid == getattr(tcp6_proc, "pid", None):
                    check_conn(
                        p,
                        conn,
                        AF_INET6,
                        SOCK_STREAM,
                        tcp6_addr,
                        (),
                        psutil.CONN_LISTEN,
                        ("all", "inet", "inet6", "tcp", "tcp6"),
                    )
                # UDP v6
                elif p.pid == getattr(udp6_proc, "pid", None):
                    check_conn(
                        p,
                        conn,
                        AF_INET6,
                        SOCK_DGRAM,
                        udp6_addr,
                        (),
                        psutil.CONN_NONE,
                        ("all", "inet", "inet6", "udp", "udp6"),
                    )

    def test_count(self):
        with create_sockets():
            # tcp
            cons = this_proc_net_connections(kind='tcp')
            assert len(cons) == (2 if supports_ipv6() else 1)
            for conn in cons:
                assert conn.family in (AF_INET, AF_INET6)
                assert conn.type == SOCK_STREAM
            # tcp4
            cons = this_proc_net_connections(kind='tcp4')
            assert len(cons) == 1
            assert cons[0].family == AF_INET
            assert cons[0].type == SOCK_STREAM
            # tcp6
            if supports_ipv6():
                cons = this_proc_net_connections(kind='tcp6')
                assert len(cons) == 1
                assert cons[0].family == AF_INET6
                assert cons[0].type == SOCK_STREAM
            # udp
            cons = this_proc_net_connections(kind='udp')
            assert len(cons) == (2 if supports_ipv6() else 1)
            for conn in cons:
                assert conn.family in (AF_INET, AF_INET6)
                assert conn.type == SOCK_DGRAM
            # udp4
            cons = this_proc_net_connections(kind='udp4')
            assert len(cons) == 1
            assert cons[0].family == AF_INET
            assert cons[0].type == SOCK_DGRAM
            # udp6
            if supports_ipv6():
                cons = this_proc_net_connections(kind='udp6')
                assert len(cons) == 1
                assert cons[0].family == AF_INET6
                assert cons[0].type == SOCK_DGRAM
            # inet
            cons = this_proc_net_connections(kind='inet')
            assert len(cons) == (4 if supports_ipv6() else 2)
            for conn in cons:
                assert conn.family in (AF_INET, AF_INET6)
                assert conn.type in (SOCK_STREAM, SOCK_DGRAM)
            # inet6
            if supports_ipv6():
                cons = this_proc_net_connections(kind='inet6')
                assert len(cons) == 2
                for conn in cons:
                    assert conn.family == AF_INET6
                    assert conn.type in (SOCK_STREAM, SOCK_DGRAM)
            # Skipped on BSD becayse by default the Python process
            # creates a UNIX socket to '/var/run/log'.
            if HAS_NET_CONNECTIONS_UNIX and not (FREEBSD or NETBSD):
                cons = this_proc_net_connections(kind='unix')
                assert len(cons) == 3
                for conn in cons:
                    assert conn.family == AF_UNIX
                    assert conn.type in (SOCK_STREAM, SOCK_DGRAM)


@pytest.mark.skipif(SKIP_SYSCONS, reason="requires root")
class TestSystemWideConnections(ConnectionTestCase):
    """Tests for net_connections()."""

    def test_it(self):
        def check(cons, families, types_):
            for conn in cons:
                assert conn.family in families
                if conn.family != AF_UNIX:
                    assert conn.type in types_
                check_connection_ntuple(conn)

        with create_sockets():
            from psutil._common import conn_tmap

            for kind, groups in conn_tmap.items():
                # XXX: SunOS does not retrieve UNIX sockets.
                if kind == 'unix' and not HAS_NET_CONNECTIONS_UNIX:
                    continue
                families, types_ = groups
                cons = psutil.net_connections(kind)
                assert len(cons) == len(set(cons))
                check(cons, families, types_)

    @retry_on_failure()
    def test_multi_sockets_procs(self):
        # Creates multiple sub processes, each creating different
        # sockets. For each process check that proc.net_connections()
        # and psutil.net_connections() return the same results.
        # This is done mainly to check whether net_connections()'s
        # pid is properly set, see:
        # https://github.com/giampaolo/psutil/issues/1013
        with create_sockets() as socks:
            expected = len(socks)
        pids = []
        times = 10
        fnames = []
        for _ in range(times):
            fname = self.get_testfn()
            fnames.append(fname)
            src = textwrap.dedent("""\
                import time, os
                from psutil.tests import create_sockets
                with create_sockets():
                    with open(r'%s', 'w') as f:
                        f.write("hello")
                    [time.sleep(0.1) for x in range(100)]
                """ % fname)
            sproc = self.pyrun(src)
            pids.append(sproc.pid)

        # sync
        for fname in fnames:
            wait_for_file(fname)

        syscons = [
            x for x in psutil.net_connections(kind='all') if x.pid in pids
        ]
        for pid in pids:
            assert len([x for x in syscons if x.pid == pid]) == expected
            p = psutil.Process(pid)
            assert len(p.net_connections('all')) == expected


class TestMisc(PsutilTestCase):
    def test_net_connection_constants(self):
        ints = []
        strs = []
        for name in dir(psutil):
            if name.startswith('CONN_'):
                num = getattr(psutil, name)
                str_ = str(num)
                assert str_.isupper(), str_
                assert str not in strs
                assert num not in ints
                ints.append(num)
                strs.append(str_)
        if SUNOS:
            psutil.CONN_IDLE  # noqa
            psutil.CONN_BOUND  # noqa
        if WINDOWS:
            psutil.CONN_DELETE_TCB  # noqa
