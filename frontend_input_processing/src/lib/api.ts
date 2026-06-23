const DEFAULT_BACKEND_PORT = 9888
const BACKEND_PORT_KEY = 'ggd_backend_port'

export type PublicConfig = {
  server: {
    host: string
    port: number
  }
  vision: {
    mode: 'adb' | 'window' | string
    target: string
    fps_limit: number
  }
  game_setting: {
    seat_num: number
  }
}

export const getBackendPort = () => {
  const savedPort = Number(window.localStorage.getItem(BACKEND_PORT_KEY))
  return Number.isInteger(savedPort) && savedPort > 0 ? savedPort : DEFAULT_BACKEND_PORT
}

export const rememberBackendPort = (port: number) => {
  if (Number.isInteger(port) && port > 0) {
    window.localStorage.setItem(BACKEND_PORT_KEY, String(port))
  }
}

export const apiUrl = (path: string) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `http://localhost:${getBackendPort()}${normalizedPath}`
}

export const wsUrl = (path: string) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `ws://localhost:${getBackendPort()}${normalizedPath}`
}

export const apiFetch = (path: string, init?: RequestInit) => {
  return fetch(apiUrl(path), init)
}
