/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          ink: '#102A43',
          sky: '#2A9D8F',
          amber: '#F4A261',
          blush: '#E76F51',
          mist: '#E6F0F3',
        },
      },
      boxShadow: {
        soft: '0 12px 40px rgba(16, 42, 67, 0.12)',
      },
      fontFamily: {
        display: ['"Sora"', 'sans-serif'],
        body: ['"Manrope"', 'sans-serif'],
      },
      keyframes: {
        riseIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        riseIn: 'riseIn 450ms ease-out both',
      },
    },
  },
  plugins: [],
};
