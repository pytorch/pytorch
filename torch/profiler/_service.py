import socket
from .profiler import profile, tensorboard_trace_handler
from torch.autograd import (_ThreadLocalState, _ThreadLocalStateGuard, _init_kineto_TLS)
from werkzeug import exceptions, wrappers, serving, wsgi
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.shared_data import SharedDataMiddleware
from threading import Thread
from multiprocessing import current_process
import requests
import json
import time
import os
import shutil
import atexit
import logging

SLAVE_NODE = 'FALSE' if os.environ.get('SERVICE_SLAVE_NODE') is None else str(os.environ.get('SERVICE_SLAVE_NODE'))
HOST = 'localhost' if os.environ.get('SERVICE_HOST') is None else str(os.environ.get('SERVICE_HOST'))
BASE_PORT = 3181 if os.environ.get('SERVICE_BASE_PORT') is None else int(os.environ.get('SERVICE_BASE_PORT'))
MASTER_PORT = 3180 if os.environ.get('SERVICE_MASTER_PORT') is None else int(os.environ.get('SERVICE_MASTER_PORT'))
TMPLOG = './tmplog'

logger = logging.getLogger("profiler_service")
logger.setLevel(logging.INFO)

class PyTorchServiceWSGIApp(object):
    
    def __init__(self, is_master_server):
        self.prof: profile = None
        self.profiling_warmup = False
        self.profiling_started = False
        self.is_master_server = is_master_server
        if self.is_master_server:
            self.slave_urls = set()
        self.app = DispatcherMiddleware(self.not_found, {
            '/registration': self.registration_route,
            '/service': self.service_route,
        })
        self.app = SharedDataMiddleware(self.app, {
            '/log': TMPLOG
        })
    
    @wrappers.Request.application
    def registration_route(self, request):
        if request.method == 'PUT':
            if self.is_master_server:
                request_data = json.loads(request.data)
                unique_url_seq = ':'.join([str(request_data['host']), str(request_data['port']), str(request_data['pid'])])
                if request.args.get('cmd') == 'register':
                    self.slave_urls.add(unique_url_seq)
                elif request.args.get('cmd') == 'unregister':
                    self.slave_urls.remove(unique_url_seq)
                else:
                    self.bad_request(request)
                return self.respond_as_json({"success": True})
            else:
                return self.respond_as_json({"success": False})
        else:
            self.method_not_allowed(request)
    
    @wrappers.Request.application
    def service_route(self, request):
        try:
            if request.method == 'PUT':
                if request.args.get('cmd') == 'start':
                    return self.start_profiling(request)
                elif request.args.get('cmd') == 'stop':
                    return self.stop_profiling(request)
                else:
                    self.bad_request(request)
            else:
                self.method_not_allowed(request)
        except Exception as e:
            logger.error(repr(e))
            return self.respond_as_json({"success": False, "message": repr(e)})
    
    def start_profiling(self, request):
        if self.profiling_warmup or self.profiling_started:
            return self.respond_as_json({"success": False, "message": "Profiling service has already been started."})

        self.request_is_local = request.remote_addr == '127.0.0.1'
        request_data = json.loads(request.data)
        request_data['log_dir'] = TMPLOG if not self.request_is_local else request_data['log_dir']
        self.run_name = request_data['run_name']
        self.log_path = os.path.join(request_data['log_dir'], self.run_name)

        start_threads: list[Thread] = []
        if self.is_master_server:
            for unique_url_seq in self.slave_urls:
                url_arr = unique_url_seq.split(':')
                thread = Thread(target=self.start_child_profiling, args=(url_arr[0], url_arr[1], request_data,))
                start_threads.append(thread)
                thread.start()
        
        if start_threads:
            for thread in start_threads:
                thread.join()
            self.profiling_started = True
        else:
            self.prof = profile(
                on_trace_ready=tensorboard_trace_handler(self.log_path),
                record_shapes=request_data['record_shapes'],
                profile_memory=request_data['profile_memory'],
                with_stack=request_data['with_stack'],
                with_flops=request_data['with_flops']
            )
            warmup_dur = request_data['warmup_dur']
            if warmup_dur > 0:
                self.profiling_warmup = True
                thread = Thread(target=self.start_profiling_with_warmup, args=(warmup_dur, _ThreadLocalState(True), ))
                thread.start()
            else:
                self.prof.start()
                self.profiling_started = True
        
        return self.respond_as_json({"success": True, "message": "Profiling service is successfully started."})
    
    def stop_profiling(self, request):
        if self.profiling_warmup:
            return self.respond_as_json({"success": False, "message": "Profiling service is still in warmup state."})
        if not self.profiling_started:
            return self.respond_as_json({"success": False, "message": "Profiling service hasn't been started yet."})
        
        stop_threads: list[Thread] = []
        if self.is_master_server:
            self.respond_file_names = []
            for unique_url_seq in self.slave_urls:
                url_arr = unique_url_seq.split(':')
                thread = Thread(target=self.stop_child_profiling, args=(url_arr[0], url_arr[1],))
                stop_threads.append(thread)
                thread.start()

        if stop_threads:
            for thread in stop_threads:
                thread.join()
        else:
            self.prof.stop()
            self.respond_file_names = [self.prof.file_name]
        self.profiling_started = False

        return self.respond_as_json({"success": True, "message": "Profiling service is successfully accomplished.", 
            "need_log_fetch": not self.request_is_local, "file_names": self.respond_file_names})

    def start_child_profiling(self, host, port, request_data):
        baseUrl = 'http://{}:{}'.format(host, port)
        requests.put(
            url="/".join([baseUrl, "service"]), 
            json=request_data,
            params={'cmd': 'start'})
    
    def stop_child_profiling(self, host, port):
        baseUrl = 'http://{}:{}'.format(host, port)
        r = requests.put(
            url="/".join([baseUrl, "service"]),
            params={'cmd': 'stop'})
        if r.status_code == 200:
            res = r.json()
        if res['success']:
            file_names = res.pop("file_names")
            if res['need_log_fetch']:
                for file_name in file_names:
                    log_file = requests.get(url="/".join([baseUrl, "log", self.run_name, file_name]))
                    if not os.path.exists(self.log_path):
                        os.makedirs(self.log_path)
                    with open(os.path.join(self.log_path, file_name), 'w') as f:
                        f.write(log_file.text)
            self.respond_file_names.extend(file_names)
    
    def start_profiling_with_warmup(self, warmup_dur, state):
        try:
            g = _ThreadLocalStateGuard(state)
            self.prof._start_warmup()
            time.sleep(warmup_dur)
            self.prof._start_trace()
            self.profiling_started = True
        finally:
            self.profiling_warmup = False

    @wrappers.Request.application
    def bad_request(self, request):
        exceptions.abort(400)

    @wrappers.Request.application
    def not_found(self, request):
        exceptions.abort(404)
    
    @wrappers.Request.application
    def method_not_allowed(self, request):
        exceptions.abort(405)
    
    @staticmethod
    def respond_as_json(obj):
        content = json.dumps(obj)
        return wrappers.Response(content, content_type="application/json")
    
    def __call__(self, environ, start_response):
        return self.app(environ, start_response)


class Listener(object):

    def __init__(self, host: str, port: int, is_master_server: bool, master_host: str):
        self.thread = None
        self.host = host
        self.port = port
        self.master_port = MASTER_PORT
        self.is_master_server = is_master_server
        self.master_host = master_host
        self.fqdn = self.host if self.host == 'localhost' else socket.getfqdn()
        _init_kineto_TLS()
        self.state = _ThreadLocalState(True)
    
    def open(self):
        self.thread = Thread(target=self.__open, args=(), daemon=True)
        self.thread.start()

    def __open(self):
        g = _ThreadLocalStateGuard(self.state)
        while True:
            registered = False
            try:
                if not self.is_master_server:
                    registered = self.__register()
                    if not registered:
                        break
                serving.run_simple(self.host, self.port, PyTorchServiceWSGIApp(self.is_master_server))
            except OSError as e:
                if registered:
                    self.__unregister()
                if e.errno == 98:
                    self.port += 1
                else:
                    raise e
    
    def __register(self):
        registered = False
        while not registered and self.master_port < MASTER_PORT + 100:
            try:
                r = requests.put(
                    url='http://{}:{}/registration'.format(self.master_host, self.master_port), 
                    json={'host': self.fqdn, 'port': self.port, 'pid': os.getpid()},
                    params={'cmd': 'register'})
                if r.json()['success']:
                    registered = True
                else:
                    break
            except:
                self.master_port += 1
        return registered
    
    def __unregister(self):
        requests.put(
            url='http://{}:{}/registration'.format(self.master_host, self.master_port), 
            json={'host': self.fqdn, 'port': self.port, 'pid': os.getpid()}, 
            params={'cmd': 'unregister'})

if current_process().name == 'MainProcess':
    def deleteTmpLog():
        tmp_log = TMPLOG
        if os.path.exists(tmp_log):
            shutil.rmtree(tmp_log)

    atexit.register(deleteTmpLog)

    if SLAVE_NODE == 'FALSE':
        listener = Listener(HOST, MASTER_PORT, True, None)
        listener.open()
