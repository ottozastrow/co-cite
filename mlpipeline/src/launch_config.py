from fabric import Connection
src_dir = "/cluster/work/lawecon/Projects/otto_von_zastrow/masterthesis/co-cite/mlpipeline/src/"
commands = [
    "cd " + src_dir,
    "git pull",
    "sleep 10",
    "python job_generator.py",
    "sleep 10",
]

all_commands_str = "; ".join(commands)
with Connection('ovonzastro@euler.ethz.ch') as c:
    result = c.run(all_commands_str)
    msg = "Ran {.command!r} on {.connection.host}, got stdout:\n{.stdout}"
    print(msg)
