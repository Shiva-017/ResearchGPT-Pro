/** @type {import('next').NextConfig} */
const nextConfig = {
  // Only proxy API in local dev (no NEXT_PUBLIC_API_URL set)
  // In production, api.ts calls the backend directly from the browser
  async rewrites() {
    if (process.env.NEXT_PUBLIC_API_URL) {
      return [];  // production: no rewrites needed
    }
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
