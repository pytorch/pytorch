from .profiler import profile, tensorboard_trace_handler
from torch.autograd import (_ThreadLocalState, _ThreadLocalStateGuard, _init_kineto_TLS)
from werkzeug import exceptions, wrappers, serving
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


class PyTorchServiceWSGIApp(object):
    
    def __init__(self, is_main_process, main_TLS):
        self.main_TLS = main_TLS
        self.g = None
        self.prof: profile = None
        self.profiling_warmup = False
        self.profiling_started = False
        self.log_dir = None
        self.is_main_proc = is_main_process
        if self.is_main_proc:
            self.child_port = set()
        self.app = DispatcherMiddleware(self.not_found, {
            '/registration': self.registration_route,
            '/service': self.service_route,
        })
        self.app = SharedDataMiddleware(self.app, {
            '/log': './tmplog'
        })
    
    @wrappers.Request.application
    def registration_route(self, request):
        if request.method == 'PUT':
            if self.is_main_proc:
                request_data = json.loads(request.data)
                unique_port_seq = ':'.join([str(request_data['port']), str(request_data['pid'])])
                if request.args.get('cmd') == 'register':
                    self.child_port.add(unique_port_seq)
                elif request.args.get('cmd') == 'unregister':
                    self.child_port.remove(unique_port_seq)
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
            return self.respond_as_json({"success": False, "message": repr(e)})
    
    def start_profiling(self, request):
        if self.profiling_warmup or self.profiling_started:
            return self.respond_as_json({"success": False, "message": "Profiling service has already been started."})

        if self.g is None:
            self.g = _ThreadLocalStateGuard(self.main_TLS)

        request_data = json.loads(request.data)

        start_threads: list[Thread] = []
        if self.is_main_proc:
            for unique_port_seq in self.child_port:
                thread = Thread(target=self.start_child_profiling, args=(unique_port_seq.split(':')[0], request_data,))
                start_threads.append(thread)
                thread.start()
        
        if start_threads:
            for thread in start_threads:
                thread.join()
            self.profiling_started = True
        else:
            self.log_dir = request_data['log_dir']
            self.prof = profile(
                on_trace_ready=tensorboard_trace_handler(self.log_dir),
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
        if self.is_main_proc:
            for unique_port_seq in self.child_port:
                thread = Thread(target=self.stop_child_profiling, args=(unique_port_seq.split(':')[0],))
                stop_threads.append(thread)
                thread.start()

        if stop_threads:
            for thread in stop_threads:
                thread.join()
        else:
            self.prof.stop()
            log_dir = self.log_dir if not self.log_dir.startswith("./tmplog") else self.log_dir[9:]
            file_name = self.prof.file_name
        self.profiling_started = False

        return self.respond_as_json({"success": True, "message": "Profiling service is successfully accomplished.", "log_dir": log_dir, "file_name": file_name})

    def start_child_profiling(self, port, request_data):
        requests.put(
            url='http://localhost:{}/service'.format(port), 
            json=request_data,
            params={'cmd': 'start'})
    
    def stop_child_profiling(self, port):
        requests.put(
            url='http://localhost:{}/service'.format(port),
            params={'cmd': 'stop'})
    
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

    def __init__(self, host: str, port: int, is_main_proc: bool):
        self.thread = None
        self.host = host
        self.port = port
        self.main_port = PORT
        self.is_main_proc = is_main_proc
        _init_kineto_TLS()
        self.state = _ThreadLocalState(True)
    
    def open(self):
        self.thread = Thread(target=self.__open, args=(), daemon=True)
        self.thread.start()

    def __open(self):
        while True:
            registered = False
            try:
                if not self.is_main_proc:
                    registered = self.__register()
                serving.run_simple(self.host, self.port, PyTorchServiceWSGIApp(self.is_main_proc, self.state))
            except OSError as e:
                if registered:
                    self.__unregister()
                if e.errno == 98:
                    self.port += 1
                else:
                    raise e
    
    def __register(self):
        registered = False
        while not registered:
            try:
                r = requests.put(
                    url='http://localhost:{}/registration'.format(self.main_port), 
                    json={'port': self.port, 'pid': os.getpid()}, 
                    params={'cmd': 'register'})
                if r.json()['success']:
                    registered = True
                else:
                    break
            except:
                self.main_port += 1
        return registered
    
    def __unregister(self):
        requests.put(
            url='http://localhost:{}/registration'.format(self.main_port), 
            json={'port': self.port, 'pid': os.getpid()}, 
            params={'cmd': 'unregister'})
 

HOST = 'localhost' if os.environ.get('PYTORCH_PROFILER_SERVICE_HOST') is None else str(os.environ.get('PYTORCH_PROFILER_SERVICE_HOST'))
PORT = 3180 if os.environ.get('PYTORCH_PROFILER_SERVICE_PORT') is None else int(os.environ.get('PYTORCH_PROFILER_SERVICE_PORT'))

if current_process().name == 'MainProcess':
    def deleteTmpLog():
        tmp_log = './tmplog'
        if os.path.exists(tmp_log):
            shutil.rmtree(tmp_log)

    atexit.register(deleteTmpLog)

    listener = Listener(HOST, PORT, True)
    listener.open()
