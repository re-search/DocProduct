import tensorflow as tf
import collections
import re


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


class RestoreCheckpointHook(tf.compat.v1.train.SessionRunHook):
    def __init__(self,
                 init_checkpoint
                 ):
        tf.compat.v1.logging.info("Create RestoreCheckpointHook.")

        self.init_checkpoint = init_checkpoint

    def begin(self):
        tvars = tf.compat.v1.trainable_variables()
        (assignment_map, _
         ) = get_assignment_map_from_checkpoint(
            tvars, self.init_checkpoint)
        tf.compat.v1.train.init_from_checkpoint(
            self.init_checkpoint, assignment_map)

        self.saver = tf.compat.v1.train.Saver(tvars)

    def after_create_session(self, session, coord):

        pass

    def before_run(self, run_context):
        return None

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        pass
