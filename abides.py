import argparse
import importlib
import sys

if __name__ == '__main__':

  # Print system banner.
  system_name = "ABIDES: Agent-Based Interactive Discrete Event Simulation"

  print ("=" * len(system_name))
  print (system_name)
  print ("=" * len(system_name))
  print ()

  # Test command line parameters.  Only peel off the config file.
  # Anything else should be left FOR the config file to consume as agent
  # or experiment parameterization.
  parser = argparse.ArgumentParser(description='Simulation configuration.')
  parser.add_argument('-c', '--config', required=True,
                      help='Name of config file to execute')
  parser.add_argument('--config-help', action='store_true',
                    help='Print argument options for the specific config file.')

  args, config_args = parser.parse_known_args()

  # Forward only the config-specific arguments to the imported config module.
  # This preserves the historical "abides.py -c <config> -- <config args...>"
  # invocation style instead of forcing each config to re-parse the launcher args.
  if config_args and config_args[0] == '--':
    config_args = config_args[1:]
  sys.argv = [sys.argv[0]] + config_args

  # First parameter supplied is config file.
  config_file = args.config

  config = importlib.import_module('config.{}'.format(config_file),
                                   package=None)

