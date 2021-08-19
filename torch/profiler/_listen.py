from werkzeug.wrappers import Request, Response
from werkzeug.wrappers.json import JSONMixin
from werkzeug.routing import Map, Rule
from werkzeug.exceptions import HTTPException
from threading import Thread
from .profiler import profile, tensorboard_trace_handler
from torch.autograd import (_ThreadLocalState, _ThreadLocalStateGuard)


class JSONRequest(Request, JSONMixin):

    pass


class HTTPServer(object):

    def __init__(self, main_TLS: _ThreadLocalState):
        self.url_map = Map([
            Rule('/', endpoint='index'),
            Rule('/cmd/start', endpoint='start'),
            Rule('/cmd/stop', endpoint='stop')
        ])
        self.prof: profile = None
        self.g = _ThreadLocalStateGuard(main_TLS)

    def dispatch_request(self, request):
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            if request.method == 'PUT' and endpoint == "start":
                config = request.get_json()
                self.prof = profile(
                    on_trace_ready=tensorboard_trace_handler(config['log_dir']),
                    record_shapes=config['record_shapes'],
                    profile_memory=config['profile_memory'],
                    with_stack=config['with_stack'],
                    with_flops=config['with_flops']
                )
                self.prof.start()
                return Response("sucess\n")
            elif request.method == 'PUT' and endpoint == "stop":
                self.prof.stop()
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

    def __init__(self, host: str, port: int):
        self.proc = None
        self.host = host
        self.port = port
        self.state = _ThreadLocalState(True)
    
    def open(self):
        self.proc = Thread(target=self.__open, args=(), daemon=True)
        self.proc.start()

    def __open(self):
        from werkzeug.serving import run_simple
        run_simple(self.host, self.port, HTTPServer(self.state))
