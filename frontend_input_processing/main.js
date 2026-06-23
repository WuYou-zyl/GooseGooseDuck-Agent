// 使用 ESM 语法导入
import { app, BrowserWindow, ipcMain, screen } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import fs from 'fs';   // fs 用于读取配置文件
import process from 'node:process';
import iconv from 'iconv-lite';
import pc from 'picocolors';   // 导入颜色库

function fixConsoleEncoding() {
    if (process.platform === 'win32') {
        const originalLog = console.log;
        const originalError = console.error;

        console.log = function (...args) {
            // 将参数转换为字符串（处理对象和常规类型）
            const str = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' ');
            // 强行将输出转为 Windows 终端默认的 GBK 字节流
            process.stdout.write(iconv.encode(str + '\n', 'gbk'));
        };

        console.error = function (...args) {
            const str = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' ');
            process.stderr.write(iconv.encode(str + '\n', 'gbk'));
        };
    }
}

// 必须在所有 console.log 执行之前运行
fixConsoleEncoding();
// 在 ESM 中需要手动模拟 __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CENTER_NAV_ARG = '--center-navigation';
let mainWindow = null;

// 后端启动函数
let backendProcess = null;

function startBackend() {
    const isPackaged = app.isPackaged;
    
    // 通过判断前端是否跑在 Vite 开发服务器（localhost）来识别是否处于“前端联调”状态
    // 如果你用了特定的环境变量也可以用 process.env.NODE_ENV === 'development'
    const isFrontendDev = !isPackaged; 

    // ==========================================
    // 模式 1：前后端联合开发情况 (终端已手动开启 main.py)
    // 假设我们通过在启动命令加参数，或者如果你习惯手动两边开，
    // 我们可以检测一个自定义环境变量，比如 process.env.DEV_BACKEND_MANUAL
    // 这里我们用最简单实用的逻辑：如果你想让 Electron 帮你拉起源码，就走 spawn('python')；
    // 如果你已经手动在终端敲了 python main.py，这里直接 return 拦截掉，什么都不做。
    // ==========================================
    if (isFrontendDev && process.env.MANUAL_BACKEND === 'true') {
        console.log('[Electron] 检测到手动开发模式，Electron 将不自动拉起后端，请确保本地 main.py 已运行。');
        return; 
    }

    let backendExePath = '';

    if (isPackaged) {
        // ==========================================
        // 模式 3：两者均打包的情况
        // ==========================================
        backendExePath = path.join(process.resourcesPath, 'backend', 'ggd-backend.exe');
    } else {
        // ==========================================
        // 模式 2：后端已打包，前端未打包（联调情况）
        // ==========================================
        backendExePath = path.resolve(__dirname, '../dist/ggd-backend/ggd-backend.exe'); // 严格核对你的本地相对路径
    }

    console.log(`[Electron] 正在尝试拉起后端，路径: ${backendExePath}`);

    // 健壮性检查：如果配置的 .exe 路径不存在，报个错但不至于让主进程彻底挂掉
    if (!fs.existsSync(backendExePath)) {
        console.error(`❌【路径错误】未找到后端执行文件: ${backendExePath}，请检查当前开发模式！`);
        return;
    }

    // 执行子进程
    backendProcess = spawn(backendExePath, [], {
        cwd: path.dirname(backendExePath) // 极其重要：确保 _internal 依赖目录能被 exe 顺利找到
    });

    backendProcess.stdout.on('data', (chunk) => {
        const str = iconv.decode(chunk, 'gbk');
        
        if (str.includes('SUCCESS')) {
            console.log(pc.green(`[Backend LOG]: ${str.trim()}`));
        } else if (str.includes('WARNING')) {
            console.log(pc.yellow(`[Backend LOG]: ${str.trim()}`));
        } else if (str.includes('ERROR')) {
            console.log(pc.red(`[Backend LOG]: ${str.trim()}`));
        } else {
            console.log(pc.blue(`[Backend LOG]: ${str.trim()}`));
        }
    });
}
// ====================

const requestNavigationCenter = () => {
    if (mainWindow) {
        if (mainWindow.isMinimized()) mainWindow.restore();
        mainWindow.show();
        mainWindow.webContents.send('center-navigation-bar');
    }
};

const setupTaskbarTasks = () => {
    if (process.platform !== 'win32') return;

    const appPathArg = process.defaultApp ? `"${app.getAppPath()}" ` : '';
    app.setUserTasks([
        {
            program: process.execPath,
            arguments: `${appPathArg}${CENTER_NAV_ARG}`,
            title: '导航栏居中',
            description: '将悬浮导航栏移动到屏幕中间'
        }
    ]);
};

const gotSingleInstanceLock = app.requestSingleInstanceLock();

if (!gotSingleInstanceLock) {
    app.quit();
} else {
    app.on('second-instance', (_event, argv) => {
        if (argv.includes(CENTER_NAV_ARG)) {
            requestNavigationCenter();
        }
    });
}

function createWindow () {
    const win = new BrowserWindow({
        width: 800,   // 初始宽度调大一点
        height: 600,
        alwaysOnTop: true,
        transparent: true,
        frame: false,             // 保持无边框
        resizable: true,          // 允许调整窗口大小
        hasShadow: false,
        icon: path.join(__dirname, 'ggd.ico'),
        webPreferences: {
        // 确保这里的路径指向的是刚才那个 preload.js
        // 使用 path.join(__dirname, 'preload.js') 比较保险
        preload: path.join(__dirname, 'preload.js'), 
        contextIsolation: true,
        nodeIntegration: false,
        }
    });
    mainWindow = win;

    // 修改加载逻辑
    if (app.isPackaged) {
        // 🌟 生产环境：加载打包好的前端 HTML 文件
        win.loadFile(path.join(__dirname, 'dist-frontend/index.html')); 
        // ⚠️ 注意：这里的相对路径取决于你 main.js 实际被打包后的位置，
        // 如果是用 Vite 编译主进程，确保它能正确指向 dist-frontend 目录。
    } else {
        // 开发环境：加载 Vite 起的本地服务器
        win.loadURL('http://localhost:5173');
    }
    
    // 监听来自前端的最小化请求
    ipcMain.on('window-min', () => {
        win.minimize();
    }); 

    ipcMain.on('set-ignore-mouse-events', (event, ignore, options) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (win) {
            // 这里的 ignore 应该是 Vue 传过来的 true 或 false
            win.setIgnoreMouseEvents(ignore, options || { forward: true });
        }
    });

    // 窗口缩放监听
    ipcMain.on('resize-window', (event, { width, height }) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (win) {
            // setSize 的第三个参数 true 表示允许动画过渡，体验更好
            win.setSize(width, height, true); 
        }
    });

    ipcMain.on('center-navigation-bar', (event, { navCenterX }) => {
        const win = BrowserWindow.fromWebContents(event.sender);
        if (!win) return;

        const bounds = win.getBounds();
        const display = screen.getDisplayMatching(bounds);
        const workArea = display.workArea;
        const targetNavCenterX = workArea.x + Math.round(workArea.width / 2);
        const maxX = Math.max(workArea.x, workArea.x + workArea.width - bounds.width);
        const nextX = Math.min(
            Math.max(bounds.x + targetNavCenterX - Math.round(navCenterX), workArea.x),
            maxX
        );

        win.setBounds({ x: nextX, y: bounds.y, width: bounds.width, height: bounds.height }, true);
    });

    win.on('closed', () => {
        if (mainWindow === win) mainWindow = null;
    });
}

if (gotSingleInstanceLock) {
    app.whenReady().then(() => {
        setupTaskbarTasks();
        // 开启后端
        startBackend();
        createWindow();

        if (process.argv.includes(CENTER_NAV_ARG)) {
            requestNavigationCenter();
        }
    });

    app.on('window-all-closed', () => {
      if (backendProcess) backendProcess.kill();
      if (process.platform !== 'darwin') app.quit();
    });
}
