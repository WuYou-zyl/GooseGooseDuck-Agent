<script setup lang="ts">
import { ref, onMounted, onUnmounted, provide, watch } from 'vue'
import ScreenCapture from './components/ScreenCapture.vue'
import SpeechLog from './components/SpeechLog.vue'
import SettingsPanel from './components/SettingsPanel.vue'
import { wsUrl } from './lib/api'

// --- 接口定义 ---
interface IElectronAPI {
  send: (channel: string, ...args: any[]) => void;
  on: (channel: string, callback: (...args: any[]) => void) => void;
}
const electron = (window as any).electronAPI as IElectronAPI

// --- 组件引用与状态 ---
const speechLogRef = ref<InstanceType<typeof SpeechLog> | null>(null)
const screenCaptureRef = ref<InstanceType<typeof ScreenCapture> | null>(null)
const navRef = ref<HTMLElement | null>(null)

const isConnected = ref(false)
const showVision = ref(false)
const showLog = ref(true)
const showSettings = ref(false)
let socket: WebSocket | null = null

// --- 核心控制：穿透逻辑 ---
let ignoreTimer: number | null = null

const stopIgnore = () => {
  if (ignoreTimer) clearTimeout(ignoreTimer)
  electron?.send('set-ignore-mouse-events', false)
}

const startIgnore = () => {
  if (ignoreTimer) clearTimeout(ignoreTimer)
  // 如果当前鼠标就在导航栏或面板上，不执行穿透，防止“闪烁”
  ignoreTimer = window.setTimeout(() => {
    electron?.send('set-ignore-mouse-events', true, { forward: true })
  }, 100)
}

/**
 * 核心：状态检查函数
 * 根据当前 UI 显示情况，决定窗口应该是实体还是穿透
 */
const syncIgnoreState = () => {
  // 如果“视觉”和“日志”都关了，窗口应该进入穿透模式（导航栏通过 mouseenter 会自己变回实体）
  if (!showVision.value && !showLog.value && !showSettings.value) {
    startIgnore()
  } else {
    // 如果有面板开着，默认先不穿透，等鼠标移出面板时由 mouseleave 触发 startIgnore
    stopIgnore()
  }
}

// 1. 监听面板显示状态变化
watch([showVision, showLog, showSettings], () => {
  syncIgnoreState()
})

// --- WebSocket 逻辑 ---
const connectService = () => {
  if (socket && (socket.readyState <= 1)) return
  socket = new WebSocket(wsUrl('/api/v1/ws/stream'))
  socket.onopen = () => { isConnected.value = true; screenCaptureRef.value?.requestNewFrame() }
  socket.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      if (['processed_frame', 'calibration_frame'].includes(data.type)) screenCaptureRef.value?.handleVisionData(data)
      if (['new', 'update', 'processed_frame'].includes(data.type)) speechLogRef.value?.handleIncomingSpeech(data)
    } catch (err) { console.error("WS Error:", err) }
  }
  socket.onclose = () => { isConnected.value = false; socket = null }
}

const disconnectService = () => { if (socket) { socket.close(); isConnected.value = false } }
const toggleConnection = () => isConnected.value ? disconnectService() : connectService()
const sendMessage = (msg: any) => { if (socket?.readyState === 1) socket.send(JSON.stringify(msg)) }
provide('ws_context', { isConnected, sendMessage, connectService, disconnectService })

// --- 生命周期 ---
onMounted(() => {
  connectService()
  
  // 2. 关键修复：处理从最小化恢复的情况
  // 即使你最小化再打开，只要面板是关着的，就自动进入穿透
  syncIgnoreState()

  // 如果你在 main.js 中转发了窗口恢复事件，可以在这里监听
  electron?.on('center-navigation-bar', centerNavigationBar)
})
onUnmounted(() => disconnectService())

const centerNavigationBar = () => {
  const navRect = navRef.value?.getBoundingClientRect()
  if (!navRect) return

  electron?.send('center-navigation-bar', {
    navCenterX: navRect.left + navRect.width / 2
  })
}

const handleWindowResize = (contentSize: { width: number, height: number }) => {
  // 计算最终窗口尺寸
  // 注意：需要加上 UI 的额外空间，例如导航栏高度(约50px)、边距(10px * 2)等
  const NAV_HEIGHT = 60; 
  const MARGIN = 20;
  const SPEECH_LOG_WIDTH = showLog.value ? 330 : 0; // 如果日志开启，也要算进去
  const SETTINGS_WIDTH = showSettings.value ? 330 : 0;

  const targetWidth = Math.max(contentSize.width + SPEECH_LOG_WIDTH + SETTINGS_WIDTH + MARGIN, 200);
  const targetHeight = contentSize.height + NAV_HEIGHT + MARGIN;

  // 调用 Electron 的窗口调整接口
  electron?.send('resize-window', {
    width: Math.round(targetWidth),
    height: Math.round(targetHeight)
  });
};
</script>

<template>
  <div class="overlay-container">
    
    <nav 
      ref="navRef"
      class="global-controller" 
      @mouseenter="stopIgnore"
      @mouseleave="syncIgnoreState" 
    >
      <div class="drag-handle">
        <span class="drag-icon">⠿</span>
      </div>
      
      <button @click="toggleConnection" :class="['btn-conn', isConnected ? 'online' : 'offline']">
        {{ isConnected ? '断开' : '连接' }}
      </button>

      <div class="divider"></div>

      <button @click="showVision = !showVision" :class="{ active: showVision }">视觉</button>
      <button @click="showLog = !showLog" :class="{ active: showLog }">日志</button>
      <button @click="showSettings = !showSettings" :class="{ active: showSettings }">设置</button>
      
      <button class="btn-min" @click="electron?.send('window-min')">一</button>
    </nav>

    <main 
      class="floating-layout"
      @mouseleave="startIgnore"
    >
      <div 
        v-if="showVision" 
        class="resizable-box vision-box" 
        @mouseenter="stopIgnore"
      >
        <ScreenCapture ref="screenCaptureRef" @update-size="handleWindowResize"/>
      </div>
      
      <div 
        v-if="showLog" 
        class="resizable-box log-box" 
        @mouseenter="stopIgnore"
      >
        <SpeechLog ref="speechLogRef" />
      </div>

      <div
        v-if="showSettings"
        class="resizable-box settings-box"
        @mouseenter="stopIgnore"
      >
        <SettingsPanel @close="showSettings = false" />
      </div>

      <div v-if="!showVision && !showLog && !showSettings" class="ghost-sensor"></div>
    </main>
  </div>
</template>

<style>
body {
  background: transparent !important;
  margin: 0;
  overflow: hidden;
  font-family: system-ui, sans-serif;
  user-select: none;
}

.overlay-container {
  position: fixed;
  top: 10px;
  right: 10px;
  width: min-content; 
  height: min-content;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  pointer-events: auto;
}

.global-controller {
  background: rgba(30, 30, 30, 0.95);
  backdrop-filter: blur(10px);
  border: 1px solid #444;
  border-radius: 6px;
  padding: 4px 8px;
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  -webkit-app-region: none;
}

.drag-handle {
  -webkit-app-region: drag;
  width: 44px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: move;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

.drag-icon { color: #888; font-size: 18px; pointer-events: none; }

.global-controller button {
  -webkit-app-region: no-drag;
  background: #2a2a2a;
  color: #ccc;
  border: 1px solid #444;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
}

.global-controller button.active {
  background: rgba(66, 184, 131, 0.2);
  color: #42b883;
  border-color: #42b883;
}

.btn-conn.online { color: #ff4444; }
.btn-conn.offline { color: #42b883; }

.floating-layout {
  display: flex;
  flex-direction: row;
  align-items: flex-start;
  gap: 10px;
  pointer-events: none;
}

.resizable-box {
  pointer-events: auto;
  background: rgba(18, 18, 18, 0.9);
  border: 1px solid #333;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0,0,0,0.7);
}

.log-box { width: 320px; height: 450px; }
.settings-box { width: 320px; }
.vision-box { min-width: 200px; min-height: 150px; }
.ghost-sensor { width: 1px; height: 1px; pointer-events: none; }
.divider { width: 1px; height: 14px; background: #444; }
.btn-min:hover { background: #c0392b !important; color: white !important; }
</style>
