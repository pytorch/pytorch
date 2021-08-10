from werkzeug.wrappers import Request, Response
from werkzeug.routing import Map, Rule
from werkzeug.exceptions import HTTPException
from multiprocessing import Process
import signal
import os

class HTTPServer(object):

    def __init__(self, main_pid: int):
        self.url_map = Map([
            Rule('/', endpoint='index'),
            Rule('/cmd/start', endpoint='start'),
            Rule('/cmd/exit', endpoint='exit')
        ])
        self.main_pid = main_pid

    def dispatch_request(self, request):
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            if request.method == 'PUT' and endpoint == "start":
                os.kill(self.main_pid, signal.SIGUSR1)
                return Response("sucess\n")
            elif request.method == 'PUT' and endpoint == "exit":
                os.kill(self.main_pid, signal.SIGUSR2)
                return Response("sucess\n")
            else:
                return Response("Error 404\n")
        except HTTPException as e:
            return e

    def wsgi_app(self, environ, start_response):
        request = Request(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)

class Listener(object):

    def __init__(self, main_pid: int, port: int):
        self.proc = None
        self.port = port
        self.main_pid = main_pid
    
    def open(self):
        self.proc = Process(target=self.__open, args=(), daemon=True)
        self.proc.start()

    def __open(self):
        from werkzeug.serving import run_simple
        run_simple('localhost', self.port, HTTPServer(self.main_pid))
