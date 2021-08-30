from werkzeug.wrappers import Request, Response
from werkzeug.routing import Map, Rule
from werkzeug.exceptions import HTTPException, abort
from threading import Thread
from .profiler import profile, tensorboard_trace_handler
from torch.autograd import (_ThreadLocalState, _ThreadLocalStateGuard, _init_kineto_TLS)
import json
import time

class HTTPServer(object):

    def __init__(self, main_TLS):
        self.url_map = Map([
            Rule('/', endpoint='index'),
            Rule('/service', endpoint='service')
        ])
        self.prof: profile = None
        self.g = _ThreadLocalStateGuard(main_TLS)
        self.profiling_warmup = False
        self.profiling_started = False

    def dispatch_request(self, request):
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            if endpoint == 'service':
                if request.method == 'PUT':
                    if request.args.get('cmd') == 'start':
                        return self.start_profiling(request)
                    elif request.args.get('cmd') == 'stop':
                        return self.stop_profiling(request)
                    else:
                        abort(400)
                else:
                    abort(405)
            else:
                abort(404)
        except HTTPException as e:
            return e
    
    def start_profiling(self, request):
        if self.profiling_warmup or self.profiling_started:
            return self.respond_as_json({"success": False, "message": "Profiling service has already been started."})
        config = json.loads(request.data)
        self.prof = profile(
            on_trace_ready=tensorboard_trace_handler(config['log_dir']),
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
            try:
                self.prof.start()
                self.profiling_started = True
            except Exception as e:
                return self.respond_as_json({"success": False, "message": repr(e)})
        return self.respond_as_json({"success": True, "message": "Profiling service is successfully started."})
    
    def stop_profiling(self, request):
        if self.profiling_warmup:
            return self.respond_as_json({"success": False, "message": "Profiling service is still in warmup state."})
        if not self.profiling_started:
            return self.respond_as_json({"success": False, "message": "Profiling service hasn't been started yet."})
        try:
            self.prof.stop()
            self.profiling_started = False
        except Exception as e:
            return self.respond_as_json({"success": False, "message": repr(e)})
        return self.respond_as_json({"success": True, "message": "Profiling service is successfully accomplished."})

    def start_profiling_with_warmup(self, warmup_dur, state):
        try:
            g = _ThreadLocalStateGuard(state)
            self.prof._start_warmup()
            time.sleep(warmup_dur)
            self.prof._start_trace()
            self.profiling_started = True
        finally:
            self.profiling_warmup = False

    @staticmethod
    def respond_as_json(obj):
        content = json.dumps(obj)
        return Response(content, content_type="application/json")

    def wsgi_app(self, environ, start_response):
        request = Request(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)


class Listener(object):

    def __init__(self, host: str, port: int):
        self.proc = None
        self.host = host
        self.port = port
        _init_kineto_TLS()
        self.state = _ThreadLocalState(True)
    
    def open(self):
        self.proc = Thread(target=self.__open, args=(), daemon=True)
        self.proc.start()

    def __open(self):
        from werkzeug.serving import run_simple
        run_simple(self.host, self.port, HTTPServer(self.state))
