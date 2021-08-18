from werkzeug.wrappers import Request, Response
from werkzeug.wrappers.json import JSONMixin
from werkzeug.routing import Map, Rule
from werkzeug.exceptions import HTTPException
from multiprocessing import Process
import signal
import os
from .profiler import profile, tensorboard_trace_handler


class JSONRequest(Request, JSONMixin):

    pass


class HTTPServer(object):

    def __init__(self, main_pid: int, shared_config: dict):
        self.url_map = Map([
            Rule('/', endpoint='index'),
            Rule('/cmd/start', endpoint='start'),
            Rule('/cmd/stop', endpoint='stop')
        ])
        self.main_pid = main_pid
        self.shared_config = shared_config

    def dispatch_request(self, request):
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            if request.method == 'PUT' and endpoint == "start":
                config = request.get_json()
                self.shared_config['log_dir'] = config['log_dir']
                self.shared_config['record_shapes'] = config['record_shapes']
                self.shared_config['profile_memory'] = config['profile_memory']
                self.shared_config['with_stack'] = config['with_stack']
                self.shared_config['with_flops'] = config['with_flops']
                os.kill(self.main_pid, signal.SIGUSR1)
                return Response("sucess\n")
            elif request.method == 'PUT' and endpoint == "stop":
                os.kill(self.main_pid, signal.SIGUSR2)
                return Response("sucess\n")
            else:
                return Response("Error 404\n")
        except HTTPException as e:
            return e

    def wsgi_app(self, environ, start_response):
        request = JSONRequest(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)


class Listener(object):

    def __init__(self, main_pid: int, host: str, port: int, shared_config: dict):
        self.proc = None
        self.host = host
        self.port = port
        self.main_pid = main_pid
        self.shared_config = shared_config
    
    def open(self):
        self.proc = Process(target=self.__open, args=(), daemon=True)
        self.proc.start()

    def __open(self):
        from werkzeug.serving import run_simple
        run_simple(self.host, self.port, HTTPServer(self.main_pid, self.shared_config))
