import subprocess
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, Button
from textual.scroll_view import ScrollView
from textual.strip import Strip
from rich.segment import Segment

class CommandOutput(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lines = []
        
    def render_line(self, y):
        if y < len(self.lines):
            return Strip([Segment(self.lines[y])])
        return Strip([Segment("")])
    
    def get_content_height(self, container, viewport):
        return len(self.lines)
    
    def add_output(self, text):
        self.lines.extend(text.split('\n'))
        self.refresh()

class MyApp(App):
    def compose(self):
        with Horizontal():
            yield Static("", classes="sidebar")
            with Vertical():
                yield Input(id="cmd")
                yield Button("Run", id="run")
                yield CommandOutput(id="output")
    
    def on_button_pressed(self):
        cmd = self.query_one("#cmd").value
        output = self.query_one("#output")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output.add_output(result.stdout + result.stderr)

MyApp().run()