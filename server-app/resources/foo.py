from flask_restful import Resource

class Foo(Resource):
    def get(self):
        return "foo"
