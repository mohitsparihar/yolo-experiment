"use client"
import { useState, useCallback } from "react"
import { detectEyewear, cropUrl, DetectResult, Product } from "@/lib/eyewear"

export default function EyewearDetector() {
  const [result, setResult]   = useState<DetectResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const handleFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      setResult(await detectEyewear(file))
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return (
    <div>
      <input type="file" accept="image/*" onChange={handleFile} disabled={loading} />
      {loading && <p>Detecting sections...</p>}
      {error   && <p style={{ color: "red" }}>{error}</p>}
      {result  && <ResultsView result={result} />}
    </div>
  )
}

function ResultsView({ result }: { result: DetectResult }) {
  return (
    <div>
      <p>Type: <strong>{result.image_type}</strong> — {result.products.length} product(s) found</p>
      {result.products.map(p => <ProductCard key={p.product_index} product={p} />)}
    </div>
  )
}

function ProductCard({ product }: { product: Product }) {
  return (
    <div style={{ border: "1px solid #ccc", padding: 12, marginTop: 12 }}>
      <p><strong>Product {product.product_index}</strong></p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {Object.entries(product.sections).map(([label, section]) => (
          <div key={label} style={{ textAlign: "center" }}>
            <img
              src={cropUrl(section.crop_url)}
              alt={label}
              style={{ width: 100, height: 80, objectFit: "cover", borderRadius: 4 }}
            />
            <p style={{ fontSize: 11, margin: "4px 0 0" }}>
              {label}<br />
              <span style={{ color: "#888" }}>{(section.confidence * 100).toFixed(0)}%</span>
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
