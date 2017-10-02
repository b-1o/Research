#coding:utf-8
import six

if six.PY2:
    import CGIHTTPServer
    CGIHTTPServer.test()
else:
    import http.server
    http.server.test(HandlerClass=http.server.CGIHTTPRequestHandler)
