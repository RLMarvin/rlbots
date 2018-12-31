from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.class_importer import load_external_class
import keyboard

Atba2 = load_external_class("Atba/atba2.py", BaseAgent)[0]
Dweller = load_external_class("Dweller/dweller.py", BaseAgent)[0]
RashBot = load_external_class("rashBot/Agent.py", BaseAgent)[0]
Leaf = load_external_class("Leaf/Agent.py", BaseAgent)[0]
Stick = load_external_class("Stick/Agent.py", BaseAgent)[0]


class CompositeBot(BaseAgent):

    def __init__(self, name, team, index):
        super(CompositeBot, self).__init__(name, team, index)
        self.bot1 = Stick(name, team, index)
        self.bot2 = Dweller(name, team, index)
        self.bot3 = Leaf(name, team, index)
        self.bot4 = RashBot(name, team, index)
        self.bot5 = Atba2(name, team, index)
        self.bot_used = self.bot1

    def initialize_agent(self):
        keyboard.on_press_key('1', self.switch_bot)
        keyboard.on_press_key('2', self.switch_bot)
        keyboard.on_press_key('3', self.switch_bot)
        keyboard.on_press_key('4', self.switch_bot)
        keyboard.on_press_key('5', self.switch_bot)
        self.register_bot_functions(self.bot1)
        self.bot5.initialize_agent()
        self.render_bot_used_name()

    def register_bot_functions(self, bot):
        bot.renderer = self.renderer
        bot.get_field_info = self.get_field_info
        bot.send_quick_chat = self.send_quick_chat
        bot.get_ball_prediction_struct = self.get_ball_prediction_struct

    def switch_bot(self, key):
        try:
            exec("self.bot_used = self.bot" + key.name)
        except (AttributeError, SyntaxError):
            self.logger.info(f"Invalid key switch: {key.name}")
        self.render_bot_used_name()

    def render_bot_used_name(self):
        bot_name = type(self.bot_used).__name__
        self.renderer.begin_rendering()
        self.renderer.draw_string_2d(10, 60, 2, 2, f"Bot: {bot_name}", self.renderer.white())
        self.renderer.end_rendering()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        return self.bot_used.get_output(packet)
