import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Create a new figure
fig, ax = plt.subplots(figsize=(12, 8))

# Draw rectangles and add text for each flowchart step
# 1. Starting Point
ax.add_patch(Rectangle((4.5, 8), 3, 1, edgecolor='black', facecolor='lightblue'))
ax.text(6, 8.5, "Mulai", ha='center', va='center', fontsize=12, fontweight='bold')

# 2. Observasi
ax.add_patch(Rectangle((3, 6), 3, 1, edgecolor='black', facecolor='lightyellow'))
ax.text(4.5, 6.5, "Observasi", ha='center', va='center', fontsize=10)

# 3. Studi Literatur
ax.add_patch(Rectangle((6, 6), 3, 1, edgecolor='black', facecolor='lightyellow'))
ax.text(7.5, 6.5, "Studi Literatur", ha='center', va='center', fontsize=10)

# 4. Analisis Masalah & Analisis Kebutuhan
ax.add_patch(Rectangle((4.5, 4), 4, 1, edgecolor='black', facecolor='lightcoral'))
ax.text(6.5, 4.5, "Analisis Masalah &\nAnalisis Kebutuhan", ha='center', va='center', fontsize=10)

# 5. Perancangan (Design)
ax.add_patch(Rectangle((4.5, 2), 4, 1, edgecolor='black', facecolor='lightgreen'))
ax.text(6.5, 2.5, "Perancangan (Design):\nUse Case, Activity Diagram,\nERD, UI Design", ha='center', va='center', fontsize=10)

# 6. Implementasi Sistem
ax.add_patch(Rectangle((4.5, 0), 4, 1, edgecolor='black', facecolor='lightblue'))
ax.text(6.5, 0.5, "Implementasi Sistem\n(MERN Stack & Excel)", ha='center', va='center', fontsize=10)

# 7. Web Scraping
ax.add_patch(Rectangle((0.5, 0), 4, 1, edgecolor='black', facecolor='lightpink'))
ax.text(2.5, 0.5, "Web Scraping untuk Data", ha='center', va='center', fontsize=10)

# 8. Testing
ax.add_patch(Rectangle((4.5, -2), 4, 1, edgecolor='black', facecolor='lightgrey'))
ax.text(6.5, -1.5, "Pengujian:\nBlack Box, Load Testing,\nUAT", ha='center', va='center', fontsize=10)

# 9. Analisis Hasil
ax.add_patch(Rectangle((4.5, -4), 4, 1, edgecolor='black', facecolor='lightgoldenrodyellow'))
ax.text(6.5, -3.5, "Analisis Hasil", ha='center', va='center', fontsize=10)

# 10. Kesimpulan dan Saran
ax.add_patch(Rectangle((4.5, -6), 4, 1, edgecolor='black', facecolor='lightcyan'))
ax.text(6.5, -5.5, "Kesimpulan dan Saran", ha='center', va='center', fontsize=10)

# 11. Selesai
ax.add_patch(Rectangle((4.5, -8), 3, 1, edgecolor='black', facecolor='lightblue'))
ax.text(6, -7.5, "Selesai", ha='center', va='center', fontsize=12, fontweight='bold')

# Draw connecting arrows
arrow_params = dict(arrowstyle='->', color='black', lw=1.5)
ax.annotate('', xy=(6, 8), xytext=(6, 7), arrowprops=arrow_params)
ax.annotate('', xy=(4.5, 6), xytext=(5.5, 5), arrowprops=arrow_params)
ax.annotate('', xy=(7.5, 6), xytext=(6.5, 5), arrowprops=arrow_params)
ax.annotate('', xy=(6.5, 4), xytext=(6.5, 3), arrowprops=arrow_params)
ax.annotate('', xy=(6.5, 2), xytext=(6.5, 1), arrowprops=arrow_params)
ax.annotate('', xy=(6.5, 0), xytext=(6.5, -1), arrowprops=arrow_params)
ax.annotate('', xy=(6.5, -2), xytext=(6.5, -3), arrowprops=arrow_params)
ax.annotate('', xy=(6.5, -4), xytext=(6.5, -5), arrowprops=arrow_params)
ax.annotate('', xy=(6.5, -6), xytext=(6.5, -7), arrowprops=arrow_params)
ax.annotate('', xy=(4.5, 0), xytext=(3.5, 0), arrowprops=arrow_params)

# Set limits and remove axes for clarity
ax.set_xlim(0, 10)
ax.set_ylim(-9, 9)
ax.axis('off')

# Save the diagram as a PNG file
plt.savefig('/mnt/data/flowchart_tahapan_penelitian_webscraping.png', dpi=300, bbox_inches='tight')
plt.show()