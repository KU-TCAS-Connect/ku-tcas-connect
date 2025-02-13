/** @type {import('tailwindcss').Config} */
const { colors: defaultColors } = require('tailwindcss/defaultTheme')

module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'kutcas-green-100':'#D9EBEE',
        'kutcas-green-700' : "#0097B2",
      }
    },
  },
  plugins: [],
}

