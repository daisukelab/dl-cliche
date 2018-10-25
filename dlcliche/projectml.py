from easydict import EasyDict

class ProjectML:
    """A framework class for driving Machine Learning Project."""
    def __init__(self,
                 setup_fn=None,
                 cycle_update_parameter_fn=None,
                 cycle_setup_data_fn=None,
                 cycle_train_model_fn=None,
                 cycle_evaluate_fn=None,
                 cycle_update_policy_fn=None,
                 summarize_total_fn=None,
                 dataset_policy={},
                 training_policy={},
                 parameters={}):
        """Instantiate project."""
        self.vars = EasyDict()
        self.results = EasyDict()
        self.reset(
              setup_fn,
              cycle_update_parameter_fn,
              cycle_setup_data_fn,
              cycle_train_model_fn,
              cycle_evaluate_fn,
              cycle_update_policy_fn,
              summarize_total_fn,
              dataset_policy,
              training_policy,
              parameters)
    def reset(self,
              setup_fn=None,
              cycle_update_parameter_fn=None,
              cycle_setup_data_fn=None,
              cycle_train_model_fn=None,
              cycle_evaluate_fn=None,
              cycle_update_policy_fn=None,
              summarize_total_fn=None,
              dataset_policy=None,
              training_policy=None,
              parameters=None):
        """Reset members, except vars and results."""
        self.setup_fn = setup_fn if setup_fn is not None else self.setup_fn
        self.cycle_update_parameter_fn = cycle_update_parameter_fn if cycle_update_parameter_fn is not None else self.cycle_update_parameter_fn
        self.cycle_setup_data_fn = cycle_setup_data_fn if cycle_setup_data_fn is not None else self.cycle_setup_data_fn
        self.cycle_train_model_fn = cycle_train_model_fn if cycle_train_model_fn is not None else self.cycle_train_model_fn
        self.cycle_evaluate_fn = cycle_evaluate_fn if cycle_evaluate_fn is not None else self.cycle_evaluate_fn
        self.cycle_update_policy_fn = cycle_update_policy_fn if cycle_update_policy_fn is not None else self.cycle_update_policy_fn
        self.summarize_total_fn = summarize_total_fn if summarize_total_fn is not None else self.summarize_total_fn
        self.dataset_policy = EasyDict(dataset_policy) if dataset_policy is not None else self.dataset_policy
        self.training_policy = EasyDict(training_policy) if training_policy is not None else self.training_policy
        self.prms = EasyDict(parameters) if parameters is not None else self.prms
    def _call(self, fn):
        """Call function if it is valid."""
        if fn is not None:
            return fn(self)
    def setup(self, cycle=0, show_policy=False):
        """Setup project. Call once when you start."""
        self.vars._cycle = cycle
        self._call(self.setup_fn)
        if show_policy:
            print('Dataset policy: {}'.format(self.dataset_policy))
            print('Training policy: {}'.format(self.training_policy))
            print('Parameters: {}'.format(self.prms))
            print('Variables: {} variables'.format(len(self.vars)))
    def run_cycle(self):
        """Iterate one project cycle, returns False if finished."""
        print('\n[Cycle #{}]'.format(self.vars._cycle))
        self._call(self.cycle_update_parameter_fn)
        self._call(self.cycle_setup_data_fn)
        self._call(self.cycle_train_model_fn)
        self._call(self.cycle_evaluate_fn)
        cycle_in_progress = self._call(self.cycle_update_policy_fn)
        print('Finished cycle #{}.\n'.format(self.vars._cycle))
        self.vars._cycle += 1
        return cycle_in_progress
    def summary(self):
        """Summarize overall performance."""
        print('\n# Summary')
        self._call(self.summarize_total_fn)
    def is_first_cycle(self):
        """Return True if it is the first cycle."""
        return self.cycle() == 0
    def cycle(self):
        return self.vars._cycle
    def run(self, cycle=0):
        """Run all through life of this project."""
        self.setup(show_policy=True, cycle=cycle)
        self.iterate()
        self.summary()
    def iterate(self):
        """Run iteration cycle."""
        while self.run_cycle():
            pass

