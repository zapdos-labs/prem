# main.py
import os
import pty
import subprocess
import threading
import queue
import signal
import shutil
from typing import Dict, List

from textual.app import App, ComposeResult
from textual.widgets import Static
from rich.text import Text


class FuckingSimpleMonitor(App):
    CSS = """
    Screen { layout: vertical; }

    /* Container for sidebar and main area */
    #content {
        layout: horizontal;
        height: 1fr;
    }

    /* Sidebar: fixed width, full height, darker background */
    #sidebar {
        width: 24;
        padding: 0;
        height: 100%;
        background: #292524;
    }

    /* Main area: remaining width, allow overflow */
    #main {
        width: 1fr;
        padding: 1;
        height: 100%;
        overflow: auto;
    }

    /* Footer: fixed height at bottom, darker background */
    #footer {
        height: 1;
        padding: 0 1;
        background: #1c1917;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("pageup", "page_up", "PageUp"),
        ("pagedown", "page_down", "PageDown"),
    ]

    def compose(self) -> ComposeResult:
        # Container for horizontal layout of sidebar and main
        with Static(id="content"):
            # Sidebar is a single Static we update manually
            self.sidebar = Static(id="sidebar")
            # Main view is a Static; we render Rich Text into it
            self.main_view = Static("Select a process", id="main")
            yield self.sidebar
            yield self.main_view
        
        # Footer with tips + line count
        self.footer = Static(id="footer")
        yield self.footer

    def on_mount(self) -> None:
        # processes to run (name, cmd)
        self.processes: List[tuple] = [
            ("SERVER", ["python", "app.py"]),
            ("WORKER", ["huey_consumer.py", "huey_worker.huey"]),
            ("TEST", ["seq", "300"]),
        ]

        # state
        self.logs: Dict[str, List[str]] = {}
        self.queues: Dict[str, queue.Queue] = {}
        self.procs: Dict[str, subprocess.Popen] = {}
        self.master_fds: Dict[str, int] = {}
        self.scroll_offsets: Dict[str, int] = {}

        # sidebar selection index
        self.selected_index: int = 0
        self.selected_name: str | None = None

        # store last computed visible range for footer display
        self._last_line_range: tuple[int, int, int] | None = None

        # spawn processes with PTY
        for name, cmd in self.processes:
            self.logs[name] = []
            self.queues[name] = queue.Queue()
            self.scroll_offsets[name] = 0

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["FORCE_COLOR"] = "1"

            master_fd, slave_fd = pty.openpty()
            proc = subprocess.Popen(
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                close_fds=True,
            )
            os.close(slave_fd)

            self.procs[name] = proc
            self.master_fds[name] = master_fd

            t = threading.Thread(
                target=self._read_from_pty, args=(name, master_fd), daemon=True
            )
            t.start()

        # initial render
        if self.processes:
            self.selected_index = 0
            self.selected_name = self.processes[0][0]
        self._render_sidebar()
        self._render_selected(self.selected_name)
        self._render_footer()

        # focus so keybindings are active
        self.set_focus(self.sidebar)

        # periodic flush
        self.set_interval(0.1, self._flush_queues_to_logs)

        # SIGINT handling so on_unmount runs
        try:
            self._orig_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._on_sigint)
        except Exception:
            self._orig_sigint = None

    def _on_sigint(self, *_):
        self.exit()

    def _read_from_pty(self, name: str, fd: int) -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    decoded = line.decode(errors="replace")
                except Exception:
                    decoded = repr(line)
                self.queues[name].put(decoded)
        if buf:
            try:
                leftover = buf.decode(errors="replace")
            except Exception:
                leftover = repr(buf)
            self.queues[name].put(leftover)

    def _flush_queues_to_logs(self) -> None:
        changed_any = False
        for name, q in self.queues.items():
            changed = False
            while not q.empty():
                line = q.get()
                if line is None:
                    continue
                self.logs[name].append(line)
                if len(self.logs[name]) > 5000:
                    self.logs[name] = self.logs[name][-5000:]
                changed = True
            if changed:
                changed_any = True
                # if currently selected and at bottom, render
                if name == self.selected_name:
                    if self.scroll_offsets.get(name, 0) == 0:
                        self._render_selected(name)
                    else:
                        # refresh without snapping
                        self._render_selected(name, keep_offset=True)
                    # footer includes line counts — keep it in sync
                    self._render_footer()
        if changed_any:
            # sidebar may show counts — re-render to keep in sync
            self._render_sidebar()

    def _render_sidebar(self) -> None:
        """Render the sidebar text with a marker for selection."""
        t = Text()
        for idx, (name, _) in enumerate(self.processes):
            prefix = "▶ " if idx == self.selected_index else "  "
            line = prefix + name
            # Pad line to fill sidebar width (24 chars) for full-line highlighting
            line = line.ljust(24)
            # style selected line
            if idx == self.selected_index:
                t.append(line + "\n", style="reverse bold")
            else:
                t.append(line + "\n")
        self.sidebar.update(t)

    def _render_footer(self) -> None:
        """Render the footer with tips from BINDINGS and the selected view's line counts."""
        t = Text()

         # Append line count info for the currently selected process
        line_info = ""
        if self.selected_name:
            # prefer last computed range (set by _render_selected) else compute quickly
            if self._last_line_range is not None:
                start, end, total = self._last_line_range
            else:
                lines = self.logs.get(self.selected_name, [])
                total = len(lines)
                offset = self.scroll_offsets.get(self.selected_name, 0)
                visible = self._visible_line_count()
                if total == 0:
                    start = 0
                    end = 0
                else:
                    end = max(1, total - offset)
                    start = max(1, end - visible + 1)
                    if start > total:
                        start = total
            if total == 0:
                line_info = f"lines: (0/0)"
            else:
                line_info = f"lines: ({start}-{end}/{total})"
            
            t.append(line_info, style="bold")
            t.append("  |  ", style="dim")



        t.append('q to quit, ↕ to cycle, PageUp/PageDown to scroll', style="dim")

       
        self.footer.update(t)

    def _visible_line_count(self) -> int:
        """Robust visible lines estimate: try textual size then fallback to terminal.
        Note: header removed, so only account for footer and a small padding.
        """
        try:
            h = getattr(self.size, "height", None) or getattr(self.size, "rows", None)
            if h:
                return max(6, int(h) - 3)  # Account for footer (header removed)
        except Exception:
            pass
        # fallback to terminal size
        try:
            return max(6, shutil.get_terminal_size().lines - 3)  # Account for footer
        except Exception:
            return 12

    def _render_selected(self, name: str | None, keep_offset: bool = False) -> None:
        if not name:
            self.main_view.update(Text("No process selected"))
            # clear last range
            self._last_line_range = None
            # footer should reflect no selection
            self._render_footer()
            return
        lines = self.logs.get(name, [])
        total = len(lines)
        visible = self._visible_line_count()
        offset = self.scroll_offsets.get(name, 0)
        if not keep_offset:
            offset = self.scroll_offsets.get(name, 0)

        # Fix line calculation to avoid 0-0 errors
        if total == 0:
            start, end = 0, 0
            window = []
        else:
            end = max(1, total - offset)
            start = max(1, end - visible + 1)
            if start > total:
                start = total
            window = lines[start-1:end]

        combined = "\n".join(window)
        
        # store last range so footer can display counts without recomputing
        self._last_line_range = (start, end, total)

        if total == 0:
            content = Text("(no lines)\n", style="dim")
        else:
            content = Text.from_ansi(combined)
        
        # Main view now contains only the log content (header removed)
        final_text = Text()
        final_text.append(content)
        
        self.main_view.update(final_text)

    # Actions for movement
    def action_move_up(self) -> None:
        if not self.processes:
            return
        self.selected_index = (self.selected_index - 1) % len(self.processes)
        self.selected_name = self.processes[self.selected_index][0]
        # reset offset to show tail when switching
        self.scroll_offsets[self.selected_name] = 0
        self._render_sidebar()
        self._render_selected(self.selected_name)
        self._render_footer()

    def action_move_down(self) -> None:
        if not self.processes:
            return
        self.selected_index = (self.selected_index + 1) % len(self.processes)
        self.selected_name = self.processes[self.selected_index][0]
        self.scroll_offsets[self.selected_name] = 0
        self._render_sidebar()
        self._render_selected(self.selected_name)
        self._render_footer()

    def action_page_up(self) -> None:
        if not self.selected_name:
            return
        page = max(1, self._visible_line_count() - 2)
        name = self.selected_name
        self.scroll_offsets[name] = min(len(self.logs.get(name, [])), self.scroll_offsets.get(name, 0) + page)
        self._render_selected(name, keep_offset=True)
        self._render_footer()

    def action_page_down(self) -> None:
        if not self.selected_name:
            return
        page = max(1, self._visible_line_count() - 2)
        name = self.selected_name
        new_offset = max(0, self.scroll_offsets.get(name, 0) - page)
        self.scroll_offsets[name] = new_offset
        self._render_selected(name, keep_offset=True)
        self._render_footer()

    def on_unmount(self) -> None:
        try:
            if self._orig_sigint is not None:
                signal.signal(signal.SIGINT, self._orig_sigint)
        except Exception:
            pass

        # graceful then force - check if attributes exist first
        if hasattr(self, 'procs'):
            for proc in list(self.procs.values()):
                try:
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGINT)
                except Exception:
                    pass
            for proc in list(self.procs.values()):
                try:
                    if proc.poll() is None:
                        proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            self.procs.clear()

        if hasattr(self, 'master_fds'):
            for fd in list(self.master_fds.values()):
                try:
                    os.close(fd)
                except Exception:
                    pass
            self.master_fds.clear()

        if hasattr(self, 'queues'):
            self.queues.clear()
        if hasattr(self, 'logs'):
            self.logs.clear()

if __name__ == "__main__":
    FuckingSimpleMonitor().run()
