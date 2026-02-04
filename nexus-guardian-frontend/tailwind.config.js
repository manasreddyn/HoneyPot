/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                nexus: {
                    900: '#0a0a0f',
                    800: '#13131f',
                    700: '#1c1c2e',
                    accent: '#00f0ff',
                    fraud: '#ff003c',
                    safe: '#00ff9d',
                    warning: '#ffd600'
                }
            },
            fontFamily: {
                mono: ['Fira Code', 'monospace'],
                sans: ['Inter', 'sans-serif']
            },
            animation: {
                'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'spin-slow': 'spin 3s linear infinite',
            }
        },
    },
    plugins: [],
}
