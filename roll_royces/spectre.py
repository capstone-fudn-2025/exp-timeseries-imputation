import os
import itertools
import datetime
import docker
import time
import uuid

SPECTRE_CONFIG = {
    "containers": 5,
    "build_image": "ts-cullinan",
    "auto_remove": False,
    "use_cuda": True,
    "params": [
        {
            "dataset": ["PhuLien"],
            "missing_percentage": [6, 12, 18, 24, 36],
            "seed": [23844, 7645, 4504, 3420, 53260],
            "batch_size": [4],
            "models": ["lr,ada,bag,dt,svm"],
            "combination_mode": ["meow", "data_per"]
        },
        {
            "dataset": ["CuaOng"],
            "missing_percentage": [7, 14, 30, 90, 180],
            "seed": [2843, 5543, 2818, 2025, 9999],
            "batch_size": [4],
            "models": ["ada,bag,dt,svm,lr,cnn1d"],
            "combination_mode": ["meow", "data_per"]
        },
        {
            "dataset": ["HaNoi"],
            "missing_percentage": [8, 16, 24, 40, 56],
            "seed": [23844, 7984, 2025, 9999, 4355],
            "batch_size": [4],
            "models": ["ada,cnn1d,attention,multi_attention"],
            "combination_mode": ["meow", "data_per"]
        },
        {
            "dataset": ["BaTriTemp"],
            "missing_percentage": [8, 16, 24, 40, 56],
            "seed": [6479, 2843, 5388, 9485, 1857],
            "batch_size": [4],
            "models": ["knn,rf"],
            "combination_mode": ["meow", "data_per"]
        },
        {
            "dataset": ["BaTriHumidity"],
            "missing_percentage": [8, 16, 24, 40, 56],
            "seed": [2843, 5543, 2818, 2025, 9999],
            "batch_size": [4],
            "models": ["knn,rf"],
            "combination_mode": ["meow", "data_per"]
        },
    ],
    "result_name_keys": ['dataset', 'missing_percentage', 'seed', 'combination_mode'],
}


def __get_docker_command(image_name: str, auto_remove: bool, use_cuda: bool, save_dir: str, params: dict) -> str:
    """
    Get the docker command.
    """
    run_config = {
        "image": image_name,
        "name": f"cullinan_{uuid.uuid4().hex[:5]}",
        "volumes": {
            f"{os.path.join(os.getcwd(), 'data')}": {"bind": "/data", "mode": "ro"},
            f"{os.path.join(os.getcwd(), 'roll_royces', 'cullinan.py')}": {"bind": "/cullinan.py", "mode": "ro"},
            f"{os.path.join(os.getcwd(), 'modules')}": {"bind": "/modules", "mode": "ro"},
            f"{save_dir}": {"bind": "/result", "mode": "rw"}
        },
        "detach": True
    }
    if use_cuda:
        run_config["device_requests"] = [
            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        ]
    if auto_remove:
        run_config["remove"] = True

    commands = ["python", "cullinan.py"]
    for key, value in params.items():
        commands.append(f"--{key}")
        commands.append(str(value))
    run_config["command"] = commands

    return run_config


def __get_results_directory_param(keys: str, values: str, allow_keys: list[str]) -> str:
    """
    Get the parameter as name for the results directory.
    """
    return '_'.join([str(value) for key, value in zip(keys, values) if key in allow_keys])


def __print_container_config(config: dict):
    """
    Print the container configuration.
    """
    _conf = config['command'][2:]
    _conf = ' '.join(
        [f'{_conf[i].replace("--", "")}=\033[01m{_conf[i+1]}\033[0m' for i in range(0, len(_conf), 2)])
    print(f"🚀 Running \033[01m{config['name']}\033[0m | {_conf}")


def main(config: dict):
    containers = config.get('containers', 5)
    image_name = config.get('build_image', 'ts-cullinan')
    auto_remove = config.get('auto_remove', True)
    use_cuda = config.get('use_cuda', True)
    params: list[dict[str, list]] = config.get('params', [])
    result_name_keys = config.get('result_name_keys', [])

    # Get command combinations
    commands = []
    for param in params:
        # Get all possible combinations
        keys = list(param.keys())
        values = list(param.values())
        combination = list(itertools.product(*values))

        # Get docker command
        for c in combination:
            result_dir = f"{os.getcwd()}/results/{datetime.datetime.now().strftime('%y%m%d%H%M%S')}_{__get_results_directory_param(keys, c, result_name_keys)}"
            command = __get_docker_command(
                image_name, auto_remove, use_cuda, result_dir, dict(zip(keys, c)))
            commands.append(command)

    # Run containers
    print(f"🐳 Running {len(commands)} containers")
    docker_client = docker.from_env()
    try:
        while True:
            running_containers = docker_client.containers.list(
                filters={"status": "running"})
            running_containers = [
                container for container in running_containers if container.name.startswith('cullinan_')]

            if len(running_containers) == 0 and len(commands) == 0:
                print("🏁 Finished")
                break

            if len(running_containers) < containers:
                for _ in range(containers - len(running_containers)):
                    if len(commands) != 0:
                        command = commands.pop(0)
                        __print_container_config(command)
                        docker_client.containers.run(**command)

            time.sleep(1)
    except docker.errors.APIError as e:
        print(f"❌ Error: {e}. Did docker daemon running?")


if __name__ == '__main__':
    main(SPECTRE_CONFIG)
