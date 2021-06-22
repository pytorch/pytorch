if "__torch_package__" in dir():

    def is_from_package():
        return True


else:

    def is_from_package():
        return False
