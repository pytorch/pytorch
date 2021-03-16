import torchvision
import copy
from torch.package import PackageExporter, PackageImporter

# set model to whatever model you're trying to import
model = torchvision.models.resnet18()

round_trip_model = None

# originally this list is empty, add the newly found mock exclusions
# here to avoid having to re-mock them (printed out during run of script)
prev_exclusions = []

mock_exclusions = copy.deepcopy(prev_exclusions)

i = 0
# limit is how many mocked items to try to add on the next run
limit = 200
while i < limit:
    try:
        with PackageExporter("mock_script_package.pt", verbose=True) as exporter:
            # This will link instead of package source code, creates backward
            # compatability dependences on all externed mods
            exporter.extern(["torch", "sys"])

            # This will replace modules with mock implementation, used to get
            # rid of dependencies that aren't used. Excludes the items in
            # 'prev_exclusions' as well as the newly added items to 'mock_exclusions'.
            exporter.mock(
                include="**", exclude=mock_exclusions,
            )

            exporter.save_pickle("model", "model.pkl", model)

        # try to import package to turn up any missing dependencies / externs
        pack = PackageImporter("mock_script_package.pt")
        import_model = pack.load_pickle("model", "model.pkl")

        # if hit, importing was successful!
        round_trip_model = import_model
        print("Success in importing the model!")
        i = limit

    except NotImplementedError as err:
        # _mock.py will throw an NotImplementeed error when it tries to
        # access a mocked object, catch the exception and add the mocked
        # item to the list of items to not mock.
        if len(err.args) > 1 and err.args[1] == "mocked":
            print(err.args)
            print(
                f"NotImplementedError module to be excluded from mocking: {err.args[2]}"
            )
            needed_mocked_module = ".".join(err.args[2].split(".")[1:-1])
            print(needed_mocked_module)
            if err.args[3] == "MOCKED OBJECT":
                mock_exclusions.append(needed_mocked_module + ".*")
            elif err.args[3] == "MOCKED METHOD":
                mock_exclusions.append(needed_mocked_module)
        else:
            # Can hit dependencies which cannot be added to the mock exclusion
            # list (such as .so files), either add them to the extern list or refactor
            # code to fix/remove dependency.
            # If assertions are hit in importing code, that will also cause a 
            # halt in the importing process and will fail to trigger this.
            # Read stack trace to find the problematic code. 
            print(
                "Hit error other than mock not implemented error! Add dependency"
                " to extern list or refactor code to fix/remove dependency."
            )
            raise
    finally:
        print(f"all mocked so far: {len(mock_exclusions)}")
        print(mock_exclusions)
        newly_mocked = [z for z in mock_exclusions if z not in prev_exclusions]
        # add this printed list of newly_mocked items to
        # 'prev_exclusions' up top to speed up dependency resolution
        print(f"newly mocked: {len(newly_mocked)}")
        print(newly_mocked)
    i += 1

# user TODO: test the original model's output to the output
# of the round_trip model to ensure correctness
assert round_trip_model is not None

print("Done!")
