import tkinter as tk
from tkinter import ttk
import win32gui
import time


class WindowSelector:
    def __init__(self):
        self.selected_hwnd = None
        self.selected_title = None
        self.root = None

    def _enum_windows_callback(self, hwnd, window_list):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and title not in ["Program Manager", ""]:
                window_list.append((hwnd, title))
        return True

    def _get_window_list(self):
        windows = []
        win32gui.EnumWindows(self._enum_windows_callback, windows)
        return windows

    def _on_select(self, event=None):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            self.selected_title = item["values"][1]
            self.selected_hwnd = item["values"][0]
            self.root.destroy()

    def _on_refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._populate_list()

    def _populate_list(self):
        windows = self._get_window_list()
        for hwnd, title in windows:
            self.tree.insert("", "end", values=(hwnd, title))

    def _highlight_window(self, hwnd):
        try:
            rect = win32gui.GetWindowRect(hwnd)
            for _ in range(3):
                win32gui.DrawFocusRect(win32gui.GetDC(0), rect)
                time.sleep(0.1)
                win32gui.InvalidateRect(0, rect, True)
                time.sleep(0.1)
        except Exception:
            pass

    def _on_double_click(self, event):
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            hwnd = item["values"][0]
            self._highlight_window(hwnd)
            self._on_select()

    def show_dialog(self):
        self.root = tk.Tk()
        self.root.title("选择要监控的游戏窗口")
        self.root.geometry("600x400")
        self.root.resizable(True, True)

        tk.Label(self.root, text="双击选择要监控的窗口", font=("Arial", 12)).pack(pady=10)

        columns = ("hwnd", "title")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("hwnd", text="窗口句柄")
        self.tree.heading("title", text="窗口标题")
        self.tree.column("hwnd", width=100)
        self.tree.column("title", width=450)

        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=5)
        scrollbar.pack(side="right", fill="y", pady=5, padx=(0, 10))

        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<Return>", lambda e: self._on_select())

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="刷新列表", command=self._on_refresh).pack(side="left", padx=5)
        tk.Button(btn_frame, text="选择", command=self._on_select).pack(side="left", padx=5)
        tk.Button(btn_frame, text="取消", command=lambda: self.root.destroy()).pack(side="left", padx=5)

        self._populate_list()
        self.root.mainloop()
        return self.selected_hwnd, self.selected_title


def select_window():
    selector = WindowSelector()
    return selector.show_dialog()

