import yaml
from src.training import training


def load_hyper_parameters(file):
    with open(file, mode='r', encoding='utf-8') as fd:
        data = yaml.load(fd, Loader=yaml.FullLoader)
    return data


if __name__ == '__main__':
    parameters_file = 'parameters.yml'
    hyper_parameters = load_hyper_parameters(parameters_file)
    config = 'Laptop'
    parameters = hyper_parameters[config]
    print('model training hyper-parameters: {}'.format(parameters))
    training(parameters)