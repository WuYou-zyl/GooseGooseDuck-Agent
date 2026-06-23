<template>
  <div class="speech-log-panel">
    <div class="log-header">
      <div class="header-title">
        <span class="icon">💬</span>
        <span>实时发言记录</span>
      </div>
      <button class="btn-clear" @click="clearLogs">清空日志</button>
    </div>

    <div class="vision-status-bar" :class="{ 'active': currentVisionSpeaker.seat_id }">
      <div class="status-indicator">
        <span class="pulse-dot" v-if="currentVisionSpeaker.seat_id"></span>
        <span class="status-text">
          视觉锁定：
          <b v-if="currentVisionSpeaker.seat_id">
            {{ currentVisionSpeaker.seat_id }}号 
            <span class="name-tag" v-if="currentVisionSpeaker.name">({{ currentVisionSpeaker.name }})</span>
          </b>
          <i v-else>扫描中...</i>
        </span>
      </div>
    </div>
    
    <div class="log-scroll-area">
      <transition-group name="list">
        <div v-for="(log, index) in logs" :key="log.id || index" class="log-card">
          <div class="log-meta">
            <span class="seat-num">{{ log.seat_id }}号</span>
            <span class="player-name">{{ log.name || '未知玩家' }}</span>
            <span class="log-time">{{ formatTime(log.timestamp) }}</span>
          </div>
          <div class="log-content">
            {{ log.content }}
          </div>
        </div>
      </transition-group>
      
      <div v-if="logs.length === 0" class="empty-state">
        <div class="empty-icon">📂</div>
        <p>暂无语音记录</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

// 界面状态
const currentVisionSpeaker = ref({ name: '', seat_id: null as number | null });
const logs = ref<any[]>([]);

/**
 * 处理由 App.vue 分发过来的语音/视觉消息
 */
const handleIncomingSpeech = (data: any) => {
  // 1. 处理视觉检测到的发言人状态（来自后端实时轮询）
  if (data.type === 'processed_frame') {
    currentVisionSpeaker.value = {
      name: data.speaker_name || '',
      seat_id: data.active_seat
    };
    return;
  }

  // 2. 处理新的语音识别记录（ASR 完成一段话）
  if (data.type === 'new') {
    logs.value.unshift({ ...data }); 
    if (logs.value.length > 50) logs.value.pop(); // 限制记录数量防止卡顿
  } 
  
  // 3. 处理流式更新（ASR 过程中的中间结果）
  else if (data.type === 'update' && logs.value.length > 0) {
    // 假设更新的是最近的一条记录（通常 ASR 更新只针对当前正在说话的人）
    if (logs.value[0].seat_id === data.seat_id) {
        logs.value[0].content = data.content;
    }
  }
};

const clearLogs = () => {
  logs.value = [];
};

const formatTime = (ts: number) => {
  if (!ts) return '';
  const date = new Date(ts * 1000);
  return date.toLocaleTimeString('zh-CN', { 
    hour12: false, 
    hour: '2-digit', 
    minute: '2-digit', 
    second: '2-digit' 
  });
};

// 必须暴露此方法，父组件 App.vue 才能通过 ref 调用
defineExpose({ handleIncomingSpeech });
</script>

<style scoped>
.speech-log-panel { 
  width: 100%; 
  height: 100%;
  display: flex; 
  flex-direction: column; 
  background: #121212; 
  color: #eee;
}

/* 头部 */
.log-header { 
  padding: 12px 15px; 
  background: #1a1a1a; 
  display: flex; 
  justify-content: space-between; 
  align-items: center; 
  border-bottom: 1px solid #333;
}
.header-title { display: flex; align-items: center; gap: 8px; font-weight: bold; font-size: 14px; color: #42b883; }
.btn-clear { 
  background: #2a2a2a; color: #888; border: 1px solid #333; 
  padding: 3px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; 
}
.btn-clear:hover { background: #333; color: #ff4444; border-color: #ff4444; }

/* 视觉锁定状态栏 */
.vision-status-bar {
  background: #1a1a1a;
  padding: 8px 15px;
  border-bottom: 1px solid #222;
  transition: all 0.4s ease;
}
.vision-status-bar.active {
  background: rgba(66, 184, 131, 0.1);
}
.status-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: #888;
}
.status-text b { color: #42b883; }
.name-tag { font-weight: normal; color: #aaa; margin-left: 4px; }

.pulse-dot {
  width: 6px;
  height: 6px;
  background: #42b883;
  border-radius: 50%;
  box-shadow: 0 0 8px #42b883;
  animation: blink 1.5s infinite;
}

/* 列表区 */
.log-scroll-area { flex: 1; overflow-y: auto; padding: 12px; }

/* 卡片样式 */
.log-card { 
  margin-bottom: 10px; 
  padding: 10px; 
  background: #1e1e1e; 
  border-radius: 6px; 
  border-left: 3px solid #42b883;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.log-meta { font-size: 11px; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; color: #999; }
.seat-num { background: #42b883; color: #000; padding: 0px 6px; border-radius: 3px; font-weight: bold; }
.player-name { color: #fff; font-weight: bold; }
.log-time { color: #555; margin-left: auto; font-family: monospace; }
.log-content { 
  font-size: 13px; 
  color: #ccc; 
  line-height: 1.5; 
  white-space: pre-wrap; 
  word-break: break-all; 
}

/* 空状态 */
.empty-state { 
  text-align: center; color: #333; margin-top: 60px; 
}
.empty-icon { font-size: 30px; margin-bottom: 10px; opacity: 0.2; }

/* 动画 */
.list-enter-active, .list-leave-active { transition: all 0.3s ease; }
.list-enter-from { opacity: 0; transform: translateX(20px); }

@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* 滚动条美化 */
.log-scroll-area::-webkit-scrollbar { width: 4px; }
.log-scroll-area::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>