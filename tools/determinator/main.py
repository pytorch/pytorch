def output_should_run(should_run):
    if should_run:
        print("::set-output name=should_run::true")
    else:
        print("::set-output name=should_run::false")


output_should_run(False)