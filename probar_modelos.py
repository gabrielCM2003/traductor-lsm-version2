"""Script de prueba independiente para los nuevos modelos de palabras LSM.

Prueba tanto el clasificador estático ONNX (WordClassifier) como el
reconocedor de señas dinámicas (DTWRecognizer) sin modificar senas.py.

Uso:
    python probar_modelos.py
"""
from __future__ import annotations

import numpy as np

from word_classifier import WordClassifier
from dtw_recognizer import DTWRecognizer


def probar_word_classifier() -> bool:
    print("=" * 65)
    print(" 1. PRUEBA DE CLASIFICADOR ESTÁTICO ONNX (WordClassifier)")
    print("=" * 65)

    classifier = WordClassifier.try_load()
    if classifier is None:
        print("[ERROR] No se pudo cargar WordClassifier. Verifica lsm_words.onnx y word_labels.json.")
        return False

    print(f"[OK] Modelo ONNX cargado exitosamente.")
    print(f"     Clases soportadas ({len(classifier.labels)}): {classifier.labels}")

    # Vector de prueba de 126 valores (dos manos)
    dummy_vector = np.random.randn(126).astype(np.float32)
    top_palabra, top_conf = classifier.predict(dummy_vector)
    top3 = classifier.predict_topk(dummy_vector, k=3)

    print(f"\n[Resultado de Inferencia]:")
    print(f"  -> Predicción Top-1: '{top_palabra}' (Confianza: {top_conf * 100:.2f}%)")
    print(f"  -> Top-K candidatos:")
    for rank, (palabra, conf) in enumerate(top3, 1):
        print(f"     {rank}. {palabra:15s} | Confianza: {conf * 100:6.2f}%")

    return True


def probar_dtw_recognizer() -> bool:
    print("\n" + "=" * 65)
    print(" 2. PRUEBA DE RECONOCEDOR DINÁMICO DTW (DTWRecognizer)")
    print("=" * 65)

    recognizer = DTWRecognizer.try_load()
    if recognizer is None:
        print("[ERROR] No se pudo inicializar DTWRecognizer. Verifica la carpeta datos_dinamicas/.")
        return False

    print(f"[OK] DTWRecognizer cargado exitosamente.")
    print(f"     Señas soportadas ({len(recognizer.labels)}): {recognizer.labels}")
    print(f"     Conteo de plantillas por seña: {recognizer.templates_count}")

    # Tomar la primera plantilla como base y añadirle una pequeña variación
    first_word = recognizer.labels[0]
    template = recognizer._templates[first_word][0]
    query_sequence = template + np.random.randn(*template.shape).astype(np.float32) * 0.05

    print(f"\n[Evaluando Secuencia]:")
    print(f"  -> Longitud de secuencia de prueba: {len(query_sequence)} frames de 126 valores")
    print(f"  -> Seña base utilizada para la simulación: '{first_word}'")

    # Predicción con confianza
    top1_word, top1_conf = recognizer.predict(query_sequence, return_distance=False)
    top3_confs = recognizer.predict_topk(query_sequence, k=3, return_distance=False)

    # Distancias brutas
    top3_dists = recognizer.predict_topk(query_sequence, k=3, return_distance=True)

    print(f"\n[Resultado de Alineación DTW]:")
    print(f"  -> Predicción Top-1: '{top1_word}' (Probabilidad Softmax: {top1_conf * 100:.2f}%)")
    print(f"  -> Ranking por Confianza y Distancia DTW Normalizada:")
    for rank, ((w, conf), (_, dist)) in enumerate(zip(top3_confs, top3_dists), 1):
        print(f"     {rank}. {w:15s} | Confianza: {conf * 100:6.2f}% | Distancia DTW: {dist:.4f}")

    return True


def main() -> None:
    print("\n*****************************************************************")
    print("    VERIFICACIÓN INDEPENDIENTE DE COMPONENTES LSM (PALABRAS)")
    print("*****************************************************************\n")

    ok_static = probar_word_classifier()
    ok_dynamic = probar_dtw_recognizer()

    print("\n" + "=" * 65)
    print(" RESUMEN DE PRUEBAS")
    print("=" * 65)
    print(f" - Clasificador Estático (ONNX + MLP):  {'[PASÓ]' if ok_static else '[FALLÓ]'}")
    print(f" - Reconocedor Dinámico (FastDTW):       {'[PASÓ]' if ok_dynamic else '[FALLÓ]'}")

    if ok_static and ok_dynamic:
        print("\n ¡Todos los módulos se ejecutaron y respondieron correctamente!")
    else:
        print("\n Hubo errores en las pruebas.")


if __name__ == "__main__":
    main()
