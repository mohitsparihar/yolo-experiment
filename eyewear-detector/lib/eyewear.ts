const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export interface Section {
  bbox: [number, number, number, number]
  confidence: number
  crop_url: string
}

export interface Product {
  product_index: number
  product_bbox: [number, number, number, number]
  sections: Record<string, Section>
}

export interface DetectResult {
  image_id: string
  image_type: "shelf" | "single" | "worn" | "closeup"
  crops_base_url: string
  products: Product[]
}

export async function detectEyewear(file: File): Promise<DetectResult> {
  const form = new FormData()
  form.append("file", file)
  const res = await fetch(`${API_BASE}/api/eyewear/detect`, {
    method: "POST",
    body: form,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function cropUrl(crop_url: string): string {
  return `${API_BASE}${crop_url}`
}
