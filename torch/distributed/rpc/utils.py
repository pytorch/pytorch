from contextlib import contextmanager
   
@contextmanager
def group_membership_management(store, name):
    token_key = "RpcGroupManagementToken"
    my_token = f"Token{name}"
    while True:
        # Retrieve token from store to signal start of rank join/leave critical section
        print(f"{name} : {my_token}")
        returned = store.compare_set(token_key, "", my_token).decode()
        print(f"{name} : {returned}")
        if returned == my_token:
            yield
            # Finish initialization
            break
        else:
            # Store will wait for the token to be released
            print(f"{name} wait : {returned}")
            store.wait([returned])
            # The wait has completed and the key for the worker can be deleted
            # print(f"{name} delete_key : {returned}")
            # store.delete_key(returned)

    # Update from store to signal end of rank join/leave critical section
    store.set(token_key, "")
    # Other will wait for this token to be set before they execute
    store.set(my_token, "Done")
