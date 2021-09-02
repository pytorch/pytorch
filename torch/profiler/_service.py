from .profiler import profile, tensorboard_trace_handler
from torch.autograd import (_ThreadLocalState, _ThreadLocalStateGuard, _init_kineto_TLS)
from werkzeug import exceptions, wrappers, serving
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.shared_data import SharedDataMiddleware
from threading import Thread
import json
import time
import os
import shutil
import atexit


class PyTorchServiceWSGIApp(object):
    
    def __init__(self, main_TLS):
        self.g = _ThreadLocalStateGuard(main_TLS)
        self.prof: profile = None
        self.profiling_warmup = False
        self.profiling_started = False
        self.log_dir = None
        self.app = DispatcherMiddleware(self.not_found, {
            '/service': self.service_route,
        })
        self.app = SharedDataMiddleware(self.app, {
            '/log': os.path.abspath(os.path.join(os.getcwd(), 'tmplog'))
        })
    
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
        config = json.loads(request.data)
        self.log_dir = config['log_dir']
        self.prof = profile(
            on_trace_ready=tensorboard_trace_handler(os.path.abspath(os.path.join(os.getcwd(), self.log_dir))),
            record_shapes=config['record_shapes'],
            profile_memory=config['profile_memory'],
            with_stack=config['with_stack'],
            with_flops=config['with_flops']
        )
        warmup_dur = config['warmup_dur']
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
        self.prof.stop()
        log_dir = self.log_dir if not self.log_dir.startswith("./tmplog") else self.log_dir[9:]
        file_name = self.prof.file_name
        self.profiling_started = False
        return self.respond_as_json({"success": True, "message": "Profiling service is successfully accomplished.", "log_dir": log_dir, "file_name": file_name})

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

    def __init__(self, host: str, port: int):
        self.thread = None
        self.host = host
        self.port = port
        _init_kineto_TLS()
        self.state = _ThreadLocalState(True)
    
    def open(self):
        self.thread = Thread(target=self.__open, args=(), daemon=True)
        self.thread.start()

    def __open(self):
        serving.run_simple(self.host, self.port, PyTorchServiceWSGIApp(self.state))


def deleteTmpLog():
    tmp_log = os.path.abspath(os.path.join(os.getcwd(), 'tmplog'))
    if os.path.exists(tmp_log):
        shutil.rmtree(tmp_log)

atexit.register(deleteTmpLog)

host = 'localhost' if os.environ.get('PYTORCH_PROFILER_SERVICE_HOST') is None else str(os.environ.get('PYTORCH_PROFILER_SERVICE_HOST'))
port = 3180 if os.environ.get('PYTORCH_PROFILER_SERVICE_PORT') is None else int(os.environ.get('PYTORCH_PROFILER_SERVICE_PORT'))

listener = Listener(host, port)
listener.open()
