from aiogram.fsm.state import State, StatesGroup


class EvaluationFSM(StatesGroup):
    waiting_for_name = State()
    waiting_for_model_name = State()
    waiting_for_url = State()
    waiting_for_token = State()
    waiting_for_rps_level = State()
    running_evaluation = State()
