# from abc import ABCMeta, ABC

# class OpaqueBase(ABCMeta):
#     pass
# OpaqueBase = ABCMeta
type_to_fake_type = {}


class OpaqueBase(type):
    def __instancecheck__(cls, instance):
        if cls in type_to_fake_type:
            if instance is type_to_fake_type[cls]:
                return True
        return super().__instancecheck__(instance)

    def register(cls, fake_type):
        global type_to_fake_type
        type_to_fake_type[cls] = fake_type
