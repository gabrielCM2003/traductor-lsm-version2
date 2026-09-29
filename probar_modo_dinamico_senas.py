"""Pruebas de regresion para el modo dinamico integrado en senas.py.

Historial de bugs que cubre (no solo el ciclo unico feliz):
1. predict_topk() (~4.5s contra 558 plantillas) NO debe bloquear el bucle de
   HandTrackingThread.run() (corre en un hilo aparte, ver
   _classify_dynamic_sequence en senas.py). Se verifica midiendo cuanto tarda
   cada llamada a _process_dynamic_frame() mientras hay una clasificacion en
   curso: debe seguir respondiendo en milisegundos, no segundos.
2. MULTIPLES ciclos consecutivos de "levantar mano -> señar -> bajar mano ->
   clasificar" deben funcionar igual de bien, uno tras otro.
3. Un hueco corto de mano perdida A MITAD de una seña larga (ej. la J, un
   trazo largo) NO debe partirla en dos señas -- DYN_NO_HAND_MS_TO_END (700ms
   en senas.py) es justo el ajuste para tolerar eso.
4. Si una seña nueva termina de grabarse mientras la anterior TODAVIA se esta
   clasificando, se descarta con aviso (ver _process_dynamic_frame) en vez de
   lanzar una segunda clasificacion en paralelo; despues de eso el sistema
   debe seguir funcionando normalmente.
5. Las 6 letras dinamicas (J, K, Ñ, Q, X, Z) pueden comprometerse, cada una
   con la regla de su grupo (ver dynamic_commit_decision en senas.py):
   grupo NORMAL (J, Ñ, X) por confianza >= DYN_MIN_CONF O por margen amplio
   >= DYN_NORMAL_MIN_MARGIN; grupo EXPERIMENTAL (K, Q, Z) solo por margen
   >= DYN_EXPERIMENTAL_MIN_MARGIN con DYN_EXPERIMENTAL_MIN_CONF como piso.
6. La regla de margen del grupo NORMAL se valida con margenes reales
   observados (2026-09-29) usando J: 34.5pp/24.6pp/37pp deben comprometer,
   11.1pp/2.1pp deben quedar dudosos (no comprometer).
7. Mientras se graba una seña dinamica en vivo, una mano que aparece a mitad
   de la secuencia y no formaba parte de la identidad de mano(s) del primer
   frame se debe ignorar (dejarse en ceros) al construir el vector de 126,
   en vez de tratarse como una postura de dos manos.

Las pruebas 1-3 usan a proposito letras del grupo NORMAL (J, Ñ, X) para los
ciclos de "funcionamiento normal": asi committed sigue siendo un buen
indicador de que un ciclo se completo de principio a fin. La prueba 5
ejercita especificamente el etiquetado por grupo/regla; las pruebas 6 y 7 son
pruebas puras (sin camara ni DTW) para la regla de margen del grupo normal y
para el aislamiento de la mano intrusa, respectivamente.

No usa camara real: alimenta HandTrackingThread._process_dynamic_frame() con
FrameDetections sinteticos, sustituyendo build_dynamic_feature_vector para
que devuelva vectores REALES de plantillas ya guardadas en datos_dinamicas/
(en vez de landmarks aleatorios, que nunca pasarian el umbral de confianza y
no probarian nada util).

Uso:
    python probar_modo_dinamico_senas.py
"""
from __future__ import annotations

import queue
import time

import numpy as np
from PyQt6.QtWidgets import QApplication

import senas

REALTIME_FRAME_S = 1 / 30  # ritmo realista de ~30fps: los umbrales de
                            # AutoSegmenter estan en ms reales, no en frames.
IN_PROGRESS_PREFIXES = ("...", "Grabando", "Clasificando", "Esperando", "Seña descartada")
CLASSIFY_TIMEOUT_S = 12.0  # predict_topk medido en ~4.5s con 558 plantillas


def _is_final_result(text: str) -> bool:
    """True si `text` ya es un resultado (letra o '¿letra?'), no un estado
    transitorio tipo 'Grabando...'/'Clasificando... (Ns)'/idle inicial."""
    return not any(text.startswith(p) for p in IN_PROGRESS_PREFIXES)


def _fake_hand() -> "senas.HandDetection":
    lm2d = np.random.rand(21, 2).astype(np.float32)
    lm3d = np.random.rand(21, 3).astype(np.float32)
    return senas.HandDetection(handedness="Right", confidence=0.9, landmarks_2d=lm2d, landmarks_3d=lm3d)


def _feeder_from_template(template_frames: list[np.ndarray]):
    """Sustituto de build_dynamic_feature_vector: ignora los landmarks (dummy)
    del FrameDetections de prueba y devuelve, en orden, los vectores de una
    plantilla real ya guardada."""
    it = iter(template_frames)
    state = {"last": template_frames[-1]}

    def fake(_hands: dict) -> np.ndarray:
        try:
            state["last"] = next(it)
        except StopIteration:
            pass
        return state["last"]

    return fake


def feed_hand(thread, n_frames: int) -> tuple[str, float]:
    """Empuja n_frames de 'mano presente', a ritmo realista."""
    det_mano = senas.FrameDetections(hands=[_fake_hand()])
    text, conf = "", 0.0
    for _ in range(n_frames):
        text, conf = thread._process_dynamic_frame(det_mano)
        time.sleep(REALTIME_FRAME_S)
    return text, conf


def feed_gap(thread, duration_s: float) -> tuple[str, float]:
    """Empuja frames de 'sin mano' durante duration_s (tiempo real), a ritmo
    realista. No espera a que termine una clasificacion: solo cubre el hueco."""
    det_vacia = senas.FrameDetections(hands=[])
    text, conf = "", 0.0
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        text, conf = thread._process_dynamic_frame(det_vacia)
        time.sleep(REALTIME_FRAME_S)
    return text, conf


def feed_gap_watch_for(thread, duration_s: float, needle: str) -> bool:
    """Como feed_gap, pero en vez de devolver solo el ultimo texto, revisa
    TODOS los frames del hueco y confirma si `needle` aparecio en alguno.

    Hace falta porque un evento de transicion (como el aviso de 'seña
    descartada') dura exactamente UN frame: si algo mas vuelve a cambiar el
    texto en el resto del hueco (ej. el contador 'Clasificando... (Ns)' de
    una clasificacion ajena que sigue corriendo), quedarse solo con el
    ultimo texto lo taparia."""
    det_vacia = senas.FrameDetections(hands=[])
    seen = False
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        text, _ = thread._process_dynamic_frame(det_vacia)
        if needle in text:
            seen = True
        time.sleep(REALTIME_FRAME_S)
    return seen


def wait_for_result(thread, timeout_s: float = CLASSIFY_TIMEOUT_S) -> tuple[str, float, float]:
    """Sigue empujando frames de 'sin mano' hasta obtener un resultado final
    (letra o '¿letra?'), o hasta el timeout. Devuelve tambien la llamada mas
    lenta a _process_dynamic_frame observada (para detectar bloqueos)."""
    det_vacia = senas.FrameDetections(hands=[])
    max_call_ms = 0.0
    text, conf = "", 0.0
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        text, conf = thread._process_dynamic_frame(det_vacia)
        max_call_ms = max(max_call_ms, (time.perf_counter() - t0) * 1000.0)
        if _is_final_result(text):
            break
        time.sleep(REALTIME_FRAME_S)
    return text, conf, max_call_ms


# =========================================================================== #
# 1. Hueco corto a mitad de una seña larga -> debe seguir siendo UNA sola seña
# =========================================================================== #

def test_gap_mid_sign(thread, label: str, template_frames: list[np.ndarray]) -> bool:
    print(f"\n=== Prueba 1: hueco de 400ms a mitad de la seña '{label}' ({len(template_frames)} frames) ===")
    print(f"    (DYN_NO_HAND_MS_TO_END = {senas.DYN_NO_HAND_MS_TO_END}ms: 400ms no deberia cortarla)")

    mitad = len(template_frames) // 2
    senas.build_dynamic_feature_vector = _feeder_from_template(template_frames)

    text, _ = feed_hand(thread, mitad)
    assert text == "Grabando...", f"esperaba 'Grabando...' antes del hueco, obtuve '{text}'"

    text, _ = feed_gap(thread, 0.4)  # 400ms < 700ms: NO debe disparar el fin
    print(f"    tras el hueco de 400ms -> estado: '{text}' (debe seguir 'Grabando...')")
    ok_no_corte = text == "Grabando..."

    text, _ = feed_hand(thread, len(template_frames) - mitad)
    assert text == "Grabando...", f"esperaba seguir 'Grabando...' tras retomar la mano, obtuve '{text}'"

    # DTW con Numba puede resolver en menos de lo que tarda un solo frame
    # (~33ms), asi que el estado "Clasificando..." puede no llegar a
    # observarse entre el hueco de cierre y el resultado final. Se fuerza un
    # retraso artificial para poder verificar esa transicion sin depender de
    # que tan rapido sea DTW en cada momento (ver tambien
    # test_discard_while_classifying, mismo motivo).
    original_predict_topk = thread._dtw_recognizer.predict_topk
    thread._dtw_recognizer.predict_topk = lambda seq, k=3: (time.sleep(1.0), original_predict_topk(seq, k=k))[1]
    try:
        # Ahora si, el hueco real de cierre (> DYN_NO_HAND_MS_TO_END).
        text, _ = feed_gap(thread, (senas.DYN_NO_HAND_MS_TO_END / 1000.0) + 0.15)
        print(f"    tras soltar la mano de verdad -> estado: '{text}' (se espera 'Clasificando...')")
        ok_una_sola_clasificacion = text.startswith("Clasificando")

        text, conf, max_call_ms = wait_for_result(thread)
    finally:
        thread._dtw_recognizer.predict_topk = original_predict_topk

    print(f"    resultado final: '{text}' ({conf * 100:.1f}%), llamada mas lenta: {max_call_ms:.1f}ms")

    ok = ok_no_corte and ok_una_sola_clasificacion and max_call_ms < 500
    print(f"    -> {'OK' if ok else 'FALLO'}")
    return ok


# =========================================================================== #
# 2. Dos ciclos consecutivos
# =========================================================================== #

def test_two_consecutive_cycles(thread, committed: list[str], pick: list[tuple[str, list]]) -> bool:
    print("\n=== Prueba 2: dos ciclos consecutivos (letras distintas) ===")
    ok = True
    for i, (letter, seq) in enumerate(pick, start=1):
        print(f"\n  --- ciclo {i}: '{letter}' ({len(seq)} frames) ---")
        senas.build_dynamic_feature_vector = _feeder_from_template(seq)
        feed_hand(thread, len(seq))
        feed_gap(thread, (senas.DYN_NO_HAND_MS_TO_END / 1000.0) + 0.15)
        text, conf, max_call_ms = wait_for_result(thread)
        print(f"  resultado: '{text}' ({conf * 100:.1f}%), llamada mas lenta: {max_call_ms:.1f}ms")
        matched = text == letter and max_call_ms < 500
        ok = ok and matched
        print(f"  -> {'OK' if matched else 'FALLO'}")
    print(f"\nLetras comprometidas hasta ahora via letter_committed_signal: {committed}")
    # Ambas letras de esta prueba son del grupo NORMAL (ver main()), con
    # confianza alta esperable para un self-match, asi que deberian haberse
    # comprometido de verdad, no solo clasificado bien.
    letras_esperadas = [letter for letter, _ in pick]
    ok_comprometidas = all(l in committed for l in letras_esperadas)
    if not ok_comprometidas:
        print(f"  FALLO: se esperaba que {letras_esperadas} estuvieran en committed")
    return ok and ok_comprometidas


# =========================================================================== #
# 3. Seña nueva mientras la anterior aun se clasifica -> se descarta con aviso
# =========================================================================== #

def test_discard_while_classifying(
    thread, letter_a: str, frames_a: list, letter_b: str, frames_b: list, letter_c: str, frames_c: list,
) -> bool:
    print("\n=== Prueba 3: seña nueva mientras la anterior se sigue clasificando ===")

    # predict_topk() se volvio mucho mas rapido despues de que dtw_recognizer.py
    # empezo a usar Numba (de ~4.5s con 558 plantillas a ~1s con las actuales):
    # depender de que A tarde "lo suficiente" para que B le gane el tiempo ya
    # no es confiable si la implementacion de DTW se sigue optimizando. Para
    # que esta prueba no dependa de que tan rapido sea DTW en cada momento, se
    # fuerza un retraso artificial en la clasificacion de A (encima del calculo
    # real), suficiente para que el ciclo B (mucho mas corto) termine de grabar
    # mientras A sigue en curso, pase lo que pase con la velocidad real.
    original_predict_topk = thread._dtw_recognizer.predict_topk

    def predict_topk_lento(sequence, k=3):
        time.sleep(3.0)
        return original_predict_topk(sequence, k=k)

    thread._dtw_recognizer.predict_topk = predict_topk_lento
    try:
        # Ciclo A: dispara la clasificacion (artificialmente lenta) en segundo plano.
        senas.build_dynamic_feature_vector = _feeder_from_template(frames_a)
        feed_hand(thread, len(frames_a))
        text, _ = feed_gap(thread, (senas.DYN_NO_HAND_MS_TO_END / 1000.0) + 0.15)
        print(f"  ciclo A ('{letter_a}') termino de grabar -> estado: '{text}'")
        assert text.startswith("Clasificando"), "el ciclo A deberia haber empezado a clasificar"
        assert thread._dynamic_classifying is True

        # Mientras A sigue clasificando, se hace un ciclo B corto: su contenido
        # no importa, nunca deberia llegar a clasificarse.
        frames_b_corto = frames_b[:20]
        senas.build_dynamic_feature_vector = _feeder_from_template(frames_b_corto)
        feed_hand(thread, len(frames_b_corto))
        ok_descartado = feed_gap_watch_for(
            thread, (senas.DYN_NO_HAND_MS_TO_END / 1000.0) + 0.15, "descartada"
        )
        print(f"  ciclo B ('{letter_b}') termino de grabar MIENTRAS A clasificaba "
              f"-> ¿aparecio el aviso de descarte?: {ok_descartado}")
        # Sigue habiendo como maximo UNA clasificacion en curso (la de A, no una de B).
        ok_sin_segunda_clasificacion = thread._dynamic_classifying is True

        # Esperar el resultado real de A (el unico que deberia llegar).
        text, conf, max_call_ms = wait_for_result(thread)
        print(f"  resultado que llega: '{text}' ({conf * 100:.1f}%) - deberia ser el de A ('{letter_a}')")
        ok_resultado_es_de_a = text == letter_a
    finally:
        thread._dtw_recognizer.predict_topk = original_predict_topk

    # Ciclo C (velocidad real de DTW, sin el retraso artificial): confirma que,
    # tras el descarte, el sistema sigue funcionando normal.
    senas.build_dynamic_feature_vector = _feeder_from_template(frames_c)
    feed_hand(thread, len(frames_c))
    feed_gap(thread, (senas.DYN_NO_HAND_MS_TO_END / 1000.0) + 0.15)
    text, conf, max_call_ms_c = wait_for_result(thread)
    print(f"  ciclo C ('{letter_c}') tras el descarte -> resultado: '{text}' ({conf * 100:.1f}%)")
    ok_recupera = text == letter_c

    ok = ok_descartado and ok_sin_segunda_clasificacion and ok_resultado_es_de_a and ok_recupera
    print(f"    -> {'OK' if ok else 'FALLO'}")
    return ok


# =========================================================================== #
# 5. Grupos de letras: cada una etiquetada con la regla de su grupo que la
#    comprometio (confianza / margen / experimental), o sin comprometerse.
# =========================================================================== #

def test_letter_groups(thread, committed: list[str], templates: dict) -> bool:
    print("\n=== Prueba 4: grupos NORMAL/EXPERIMENTAL, cada letra con la regla que le toca ===")
    print(f"    DYN_NORMAL_LETTERS = {senas.DYN_NORMAL_LETTERS}  "
          f"DYN_EXPERIMENTAL_LETTERS = {senas.DYN_EXPERIMENTAL_LETTERS}")

    letras_experimentales = [l for l in senas.DYN_EXPERIMENTAL_LETTERS if templates.get(l)]
    letras_normales = [l for l in senas.DYN_NORMAL_LETTERS if templates.get(l)]
    if not letras_experimentales or not letras_normales:
        print("    FALLO: hacen falta plantillas de al menos una letra normal y una experimental")
        return False

    ok = True
    for letra in letras_experimentales + letras_normales:
        seq = list(templates[letra][0])
        senas.build_dynamic_feature_vector = _feeder_from_template(seq)
        feed_hand(thread, len(seq))
        feed_gap(thread, (senas.DYN_NO_HAND_MS_TO_END / 1000.0) + 0.15)
        text, conf, _ = wait_for_result(thread)

        idle_text = thread._dynamic_idle_text
        se_comprometio = letra in committed

        if letra in senas.DYN_EXPERIMENTAL_LETTERS:
            # Solo puede comprometerse por la regla EXPERIMENTAL; si no se
            # compromete, el Estado debe seguir diciendo "modo experimental".
            correcto = (
                (se_comprometio and idle_text.startswith(letra) and "margen alto" in idle_text)
                or (not se_comprometio and "modo experimental" in idle_text)
            )
            print(f"  {letra} (experimental): clasifico '{text}' ({conf*100:.1f}%), "
                  f"comprometida={se_comprometio}, Estado='{idle_text}' -> {'OK' if correcto else 'FALLO'}")
        else:
            # Grupo normal: si se compromete, debe ser por confianza o por
            # margen amplio (etiquetado distinto en el Estado).
            correcto = (not se_comprometio) or (se_comprometio and idle_text.startswith(letra))
            print(f"  {letra} (normal): clasifico '{text}' ({conf*100:.1f}%), "
                  f"comprometida={se_comprometio}, Estado='{idle_text}' -> {'OK' if correcto else 'FALLO'}")
        ok = ok and correcto

    return ok


# =========================================================================== #
# 6. Regla de margen del grupo NORMAL con margenes reales de J (pura, sin
#    camara ni DTW: alimenta dynamic_commit_decision directamente).
# =========================================================================== #

def test_normal_group_margin_rule() -> bool:
    print("\n=== Prueba 6: regla de margen del grupo NORMAL con margenes reales de J ===")
    print(f"    DYN_MIN_CONF={senas.DYN_MIN_CONF}  DYN_NORMAL_MIN_MARGIN={senas.DYN_NORMAL_MIN_MARGIN}")

    # conf1 se fija deliberadamente por debajo de DYN_MIN_CONF para aislar la
    # via de margen (si conf1 ya alcanzara DYN_MIN_CONF, comprometeria de
    # todos modos por confianza, sin probar nada sobre el margen).
    conf1 = senas.DYN_MIN_CONF - 0.05
    casos = [
        (0.345, True, "34.5pp - debe comprometer"),
        (0.246, True, "24.6pp - debe comprometer"),
        (0.37, True, "37pp - debe comprometer"),
        (0.111, False, "11.1pp - dudoso, NO debe comprometer"),
        (0.021, False, "2.1pp - dudoso, NO debe comprometer"),
    ]
    ok = True
    for margen_real, esperado, motivo in casos:
        conf2 = max(0.0, conf1 - margen_real)
        topk = [("J", conf1), ("Ñ", conf2), ("X", 0.01)]
        should_commit, margin, rule = senas.dynamic_commit_decision(topk)
        correcto = should_commit == esperado
        print(f"  margen={margin*100:.1f}pp ({motivo}) -> should_commit={should_commit} "
              f"(esperado {esperado}), regla='{rule}' -> {'OK' if correcto else 'FALLO'}")
        ok = ok and correcto
    return ok


# =========================================================================== #
# 7. Aislar la mano que no forma parte de la identidad de la secuencia (pura,
#    inspecciona que hands_by_side llega a build_dynamic_feature_vector).
# =========================================================================== #

def test_hand_identity_filter(thread) -> bool:
    print("\n=== Prueba 7: mano intrusa a mitad de secuencia se ignora (queda en ceros) ===")

    def det_with(hands_present: list[str]) -> "senas.FrameDetections":
        dets = []
        for h in hands_present:
            lm2d = np.random.rand(21, 2).astype(np.float32)
            lm3d = np.random.rand(21, 3).astype(np.float32)
            dets.append(senas.HandDetection(
                handedness=h, confidence=0.9, landmarks_2d=lm2d, landmarks_3d=lm3d
            ))
        return senas.FrameDetections(hands=dets)

    received_hands: list[set] = []
    original_builder = senas.build_dynamic_feature_vector

    def spy_builder(hands: dict) -> np.ndarray:
        received_hands.append(set(hands.keys()))
        return original_builder(hands)

    senas.build_dynamic_feature_vector = spy_builder
    try:
        # Caso A: secuencia empieza con solo mano derecha; a mitad aparece la
        # izquierda por 3 frames (sin intencion de señar) y luego desaparece.
        received_hands.clear()
        for _ in range(5):
            thread._process_dynamic_frame(det_with(["Right"]))
        for _ in range(3):
            thread._process_dynamic_frame(det_with(["Right", "Left"]))
        for _ in range(5):
            thread._process_dynamic_frame(det_with(["Right"]))
        thread._reset_dynamic_state()

        intrusion_frames = received_hands[5:8]
        ok_intrusion_filtrada = len(intrusion_frames) == 3 and all(
            hands == {"Right"} for hands in intrusion_frames
        )
        print(f"  Caso A (empieza con 1 mano, intrusa a mitad): frames de intrusion "
              f"recibidos por build_dynamic_feature_vector = {intrusion_frames} "
              f"-> {'OK' if ok_intrusion_filtrada else 'FALLO'}")

        # Caso B: ambas manos presentes desde el primer frame -> NO se filtra
        # (podria ser una seña genuina de dos manos).
        received_hands.clear()
        for _ in range(5):
            thread._process_dynamic_frame(det_with(["Right", "Left"]))
        thread._reset_dynamic_state()

        ok_dos_manos_intactas = len(received_hands) == 5 and all(
            hands == {"Right", "Left"} for hands in received_hands
        )
        print(f"  Caso B (dos manos desde el inicio): hands recibidos = {received_hands} "
              f"-> {'OK' if ok_dos_manos_intactas else 'FALLO'}")

        ok = ok_intrusion_filtrada and ok_dos_manos_intactas
        print(f"    -> {'OK' if ok else 'FALLO'}")
        return ok
    finally:
        senas.build_dynamic_feature_vector = original_builder
        thread._reset_dynamic_state()


def main() -> int:
    app = QApplication([])
    original_feeder = senas.build_dynamic_feature_vector

    try:
        cfg = senas.AppConfig()
        thread = senas.HandTrackingThread(queue.Queue(), cfg)
        thread.set_dynamic_mode(True)

        if not thread.has_dynamic_recognizer:
            print("ERROR: no hay DTWRecognizer disponible (revisa datos_dinamicas/ y fastdtw/scipy).")
            return 1

        committed: list[str] = []
        thread.letter_committed_signal.connect(lambda l: committed.append(l))

        labels = thread.dynamic_labels
        print(f"Señas dinamicas disponibles: {labels}")
        templates = thread._dtw_recognizer._templates

        # Pruebas 1-3 (mecanica de segmentacion/ciclos): a proposito, solo
        # letras del grupo NORMAL, para que "committed" siga siendo un
        # indicador valido de que el ciclo se completo de principio a fin
        # (ver docstring del modulo). Las letras experimentales (K, Q, Z) se
        # prueban aparte en test_letter_groups.
        usable = [l for l in senas.DYN_NORMAL_LETTERS if templates.get(l)]
        if len(usable) < 3:
            print("ERROR: hacen falta al menos 3 letras del grupo NORMAL con plantillas para estas pruebas.")
            return 1

        letra_1, letra_2, letra_3 = usable[0], usable[1], usable[2]
        seq_1 = list(templates[letra_1][0])
        seq_2 = list(templates[letra_2][0])
        seq_3 = list(templates[letra_3][0])

        resultados = {}
        resultados["gap_mid_sign"] = test_gap_mid_sign(thread, letra_1, seq_1)
        resultados["two_cycles"] = test_two_consecutive_cycles(
            thread, committed, [(letra_2, seq_2), (letra_3, seq_3)]
        )
        # Para la prueba de descarte se reutilizan letras ya vistas (no importa
        # para esto que se repitan, lo que se prueba es la mecanica de descarte).
        resultados["discard_while_classifying"] = test_discard_while_classifying(
            thread, letra_1, seq_1, letra_2, seq_2, letra_3, seq_3,
        )
        resultados["letter_groups"] = test_letter_groups(thread, committed, templates)
        resultados["normal_group_margin_rule"] = test_normal_group_margin_rule()
        resultados["hand_identity_filter"] = test_hand_identity_filter(thread)

        print("\n=== RESUMEN ===")
        for nombre, ok in resultados.items():
            print(f"  {nombre}: {'OK' if ok else 'FALLO'}")

        todo_ok = all(resultados.values())
        print(f"\nRESULTADO GLOBAL: {'OK - todas las pruebas pasaron' if todo_ok else 'FALLO'}")
        return 0 if todo_ok else 1
    finally:
        senas.build_dynamic_feature_vector = original_feeder


if __name__ == "__main__":
    raise SystemExit(main())
