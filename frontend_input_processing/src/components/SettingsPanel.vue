<template>
  <section class="settings-panel">
    <header class="settings-header">
      <h2>设置</h2>
      <button class="icon-button" type="button" title="关闭" @click="$emit('close')">X</button>
    </header>

    <form class="settings-form" @submit.prevent="saveConfig">
      <fieldset :disabled="loading || saving">
        <label>
          <span>后端端口</span>
          <input v-model.number="draft.server.port" type="number" min="1" max="65535" />
        </label>

        <label>
          <span>监听地址</span>
          <input v-model.trim="draft.server.host" type="text" />
        </label>

        <label>
          <span>采集模式</span>
          <select v-model="draft.vision.mode">
            <option value="adb">ADB</option>
            <option value="window">窗口</option>
          </select>
        </label>

        <label>
          <span>采集目标</span>
          <input v-model.trim="draft.vision.target" type="text" placeholder="127.0.0.1:16384" />
        </label>

        <label>
          <span>FPS 限制</span>
          <input v-model.number="draft.vision.fps_limit" type="number" min="1" max="30" />
        </label>

        <label>
          <span>游戏人数</span>
          <input v-model.number="draft.game_setting.seat_num" type="number" min="1" max="15" />
        </label>
      </fieldset>

      <p v-if="message" :class="['message', messageType]">{{ message }}</p>

      <footer class="settings-actions">
        <button class="secondary-button" type="button" @click="loadConfig">刷新</button>
        <button class="primary-button" type="submit" :disabled="loading || saving">
          {{ saving ? '保存中' : '保存' }}
        </button>
      </footer>
    </form>
  </section>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { apiFetch, rememberBackendPort, type PublicConfig } from '@/lib/api'

defineEmits(['close'])

const loading = ref(false)
const saving = ref(false)
const message = ref('')
const messageType = ref<'success' | 'error'>('success')

const draft = reactive<PublicConfig>({
  server: {
    host: '127.0.0.1',
    port: 8000,
  },
  vision: {
    mode: 'adb',
    target: '127.0.0.1:16384',
    fps_limit: 1,
  },
  game_setting: {
    seat_num: 13,
  },
})

const applyConfig = (config: PublicConfig) => {
  draft.server.host = config.server.host
  draft.server.port = config.server.port
  draft.vision.mode = config.vision.mode
  draft.vision.target = config.vision.target
  draft.vision.fps_limit = config.vision.fps_limit
  draft.game_setting.seat_num = config.game_setting?.seat_num ?? 13
}

const showMessage = (text: string, type: 'success' | 'error' = 'success') => {
  message.value = text
  messageType.value = type
}

const loadConfig = async () => {
  loading.value = true
  showMessage('')
  try {
    const res = await apiFetch('/api/v1/config')
    const data = await res.json()
    if (data.status !== 'success') throw new Error(data.message || '读取配置失败')
    applyConfig(data.config)
  } catch (error) {
    showMessage(error instanceof Error ? error.message : '读取配置失败', 'error')
  } finally {
    loading.value = false
  }
}

const saveConfig = async () => {
  saving.value = true
  showMessage('')
  const payload: PublicConfig = {
    server: {
      host: draft.server.host,
      port: Number(draft.server.port),
    },
    vision: {
      mode: draft.vision.mode,
      target: draft.vision.target,
      fps_limit: Number(draft.vision.fps_limit),
    },
    game_setting: {
      seat_num: Number(draft.game_setting.seat_num),
    },
  }

  try {
    const res = await apiFetch('/api/v1/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = await res.json()
    if (data.status !== 'success') throw new Error(data.message || '保存配置失败')
    applyConfig(data.config)
    rememberBackendPort(data.config.server.port)
    showMessage(data.message || '配置已保存，运行时配置已更新。')
  } catch (error) {
    showMessage(error instanceof Error ? error.message : '保存配置失败', 'error')
  } finally {
    saving.value = false
  }
}

onMounted(loadConfig)
</script>

<style scoped>
.settings-panel {
  width: 320px;
  color: #eee;
  background: rgba(18, 18, 18, 0.94);
}

.settings-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-bottom: 1px solid #333;
}

h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 650;
}

.icon-button,
.secondary-button,
.primary-button {
  border: 1px solid #444;
  border-radius: 4px;
  cursor: pointer;
}

.icon-button {
  width: 26px;
  height: 26px;
  background: #242424;
  color: #ccc;
}

.settings-form {
  padding: 12px;
}

fieldset {
  border: 0;
  padding: 0;
  margin: 0;
  display: grid;
  gap: 10px;
}

label {
  display: grid;
  gap: 5px;
  font-size: 12px;
  color: #bbb;
}

input,
select {
  height: 30px;
  box-sizing: border-box;
  border: 1px solid #444;
  border-radius: 4px;
  background: #202020;
  color: #eee;
  padding: 0 8px;
}

.message {
  margin: 12px 0 0;
  font-size: 12px;
  line-height: 1.5;
}

.message.success {
  color: #42b883;
}

.message.error {
  color: #ff6b6b;
}

.settings-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 14px;
}

.secondary-button,
.primary-button {
  min-width: 64px;
  height: 30px;
}

.secondary-button {
  background: #2a2a2a;
  color: #ddd;
}

.primary-button {
  background: #42b883;
  color: #06160f;
  border-color: #42b883;
  font-weight: 650;
}
</style>
