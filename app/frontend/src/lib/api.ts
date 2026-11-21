const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

type ApiRequestOptions = Omit<RequestInit, 'body'> & { body?: unknown }

export async function apiRequest<T>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers)
  if (options.body !== undefined && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  })
  if (!response.ok) throw new Error(`API request failed with status ${response.status}`)
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}
