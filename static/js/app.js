// =========================================================
// UPDATE STATUS
// =========================================================

async function updateStatus() {

    try {

        const response = await fetch(

            "/status",

            {
                cache: "no-store"
            }
        );


        if (!response.ok) {
            return;
        }


        const data =
            await response.json();


        // =================================================
        // MODE
        // =================================================

        updateAutoMode(
            data.mode
        );


        // =================================================
        // PREDICTION
        // =================================================

        document.getElementById(
            "prediction"
        ).innerText =

            data.prediction
            || "-";


        // =================================================
        // CONFIDENCE
        // =================================================

        const confidence =
            Number(
                data.confidence || 0
            );


        document.getElementById(
            "confidence"
        ).innerText =

            confidence.toFixed(1)
            + "%";


        // =================================================
        // CONFIRMED
        // =================================================

        document.getElementById(
            "confirmed"
        ).innerText =

            data.confirmed
            || "-";


        // =================================================
        // MOTION
        // =================================================

        const motion =
            Number(
                data.motion || 0
            );


        document.getElementById(
            "motionValue"
        ).innerText =

            motion.toFixed(4);


        // =================================================
        // TEXT
        // =================================================

        updateText(
            data.text || ""
        );


        // =================================================
        // SEQUENCE
        // =================================================

        updateSequence(

            data.mode,

            Number(
                data.sequence || 0
            ),

            Number(
                data.sequence_length
                || 30
            )
        );


        // =================================================
        // STATUS MESSAGE
        // =================================================

        updateRecognitionStatus(

            data.mode,

            Boolean(
                data.waiting
            ),

            Number(
                data.sequence || 0
            ),

            Number(
                data.sequence_length
                || 30
            )
        );


    } catch (error) {

        console.error(
            "Status error:",
            error
        );

    }

}


// =========================================================
// AUTO MODE
// =========================================================

function updateAutoMode(mode) {

    const currentMode =
        document.getElementById(
            "currentMode"
        );


    const modelName =
        document.getElementById(
            "modelName"
        );


    const description =
        document.getElementById(
            "modeDescription"
        );


    if (mode === "static") {

        currentMode.innerText =
            "Đang nhận diện chữ cái";


        modelName.innerText =
            "Random Forest";


        description.innerText =
            "Bàn tay đang ổn định. Hệ thống tự sử dụng Random Forest để nhận diện chữ cái.";

    }

    else if (
        mode === "dynamic"
    ) {

        currentMode.innerText =
            "Đang nhận diện từ giao tiếp";


        modelName.innerText =
            "LSTM";


        description.innerText =
            "Phát hiện chuyển động. Hệ thống tự sử dụng LSTM để nhận diện từ giao tiếp.";

    }

    else {

        currentMode.innerText =
            "Đang xác định ký hiệu";


        modelName.innerText =
            "Đang xác định";


        description.innerText =
            "Hệ thống đang phân tích chuyển động bàn tay để chọn mô hình phù hợp.";

    }

}


// =========================================================
// SEQUENCE
// =========================================================

function updateSequence(
    mode,
    current,
    total
) {

    const container =
        document.getElementById(
            "sequenceContainer"
        );


    const sequence =
        document.getElementById(
            "sequence"
        );


    const bar =
        document.getElementById(
            "sequenceBar"
        );


    // Chỉ hiện khi Dynamic
    if (mode !== "dynamic") {

        container.style.display =
            "none";


        bar.style.width =
            "0%";


        sequence.innerText =
            "0/" + total;


        return;
    }


    container.style.display =
        "block";


    sequence.innerText =

        current
        + "/"
        + total;


    let percent = 0;


    if (total > 0) {

        percent =
            (
                current
                /
                total
            )
            * 100;
    }


    percent = Math.min(
        percent,
        100
    );


    bar.style.width =
        percent + "%";

}


// =========================================================
// RECOGNITION STATUS
// =========================================================

function updateRecognitionStatus(
    mode,
    waiting,
    sequence,
    total
) {

    const element =
        document.getElementById(
            "recognitionStatus"
        );


    // =====================================================
    // WAIT REMOVE HAND
    // =====================================================

    if (waiting) {

        element.innerText =
            "Đã nhận diện - hãy bỏ tay ra trước khi thực hiện ký hiệu tiếp theo";


        element.classList.add(
            "waiting"
        );


        return;
    }


    element.classList.remove(
        "waiting"
    );


    // =====================================================
    // STATIC
    // =====================================================

    if (mode === "static") {

        element.innerText =
            "Giữ ổn định ký hiệu chữ cái...";


        return;
    }


    // =====================================================
    // DYNAMIC
    // =====================================================

    if (mode === "dynamic") {

        if (sequence < total) {

            element.innerText =

                "Đang thu chuyển động "
                + sequence
                + "/"
                + total;

        } else {

            element.innerText =
                "Đang phân tích từ giao tiếp...";
        }


        return;
    }


    // =====================================================
    // WAITING
    // =====================================================

    element.innerText =
        "Đang phân tích chuyển động bàn tay...";

}


// =========================================================
// TEXT
// =========================================================

function updateText(text) {

    const element =
        document.getElementById(
            "resultText"
        );


    if (
        text
        &&
        text.trim() !== ""
    ) {

        element.innerText =
            text;


        element.classList.remove(
            "placeholder"
        );

    } else {

        element.innerText =
            "Kết quả sẽ xuất hiện tại đây...";


        element.classList.add(
            "placeholder"
        );

    }

}


// =========================================================
// ADD SPACE
// =========================================================

async function addSpace() {

    try {

        const response = await fetch(

            "/add_space",

            {
                method: "POST"
            }
        );


        const data =
            await response.json();


        if (data.success) {

            updateText(
                data.text || ""
            );
        }


    } catch (error) {

        console.error(
            error
        );

    }

}


// =========================================================
// DELETE
// =========================================================

async function deleteLast() {

    try {

        const response = await fetch(

            "/delete_last",

            {
                method: "POST"
            }
        );


        const data =
            await response.json();


        if (data.success) {

            updateText(
                data.text || ""
            );
        }


    } catch (error) {

        console.error(
            error
        );

    }

}


// =========================================================
// CLEAR
// =========================================================

async function clearText() {

    try {

        const response = await fetch(

            "/clear",

            {
                method: "POST"
            }
        );


        const data =
            await response.json();


        if (data.success) {

            updateText("");

            document.getElementById(
                "prediction"
            ).innerText = "-";


            document.getElementById(
                "confidence"
            ).innerText = "0%";


            document.getElementById(
                "confirmed"
            ).innerText = "-";


            document.getElementById(
                "motionValue"
            ).innerText = "0.0000";


            updateAutoMode(
                "waiting"
            );


            updateSequence(
                "waiting",
                0,
                30
            );
        }


    } catch (error) {

        console.error(
            error
        );

    }

}


// =========================================================
// TEXT TO SPEECH
// =========================================================

function speakText() {

    const element =
        document.getElementById(
            "resultText"
        );


    if (
        element.classList.contains(
            "placeholder"
        )
    ) {
        return;
    }


    const text =
        element.innerText.trim();


    if (!text) {
        return;
    }


    if (
        !(
            "speechSynthesis"
            in window
        )
    ) {

        alert(
            "Trình duyệt không hỗ trợ đọc văn bản."
        );

        return;
    }


    window.speechSynthesis.cancel();


    const speech =
        new SpeechSynthesisUtterance(
            text
        );


    speech.lang =
        "vi-VN";


    speech.rate = 1;

    speech.pitch = 1;

    speech.volume = 1;


    window.speechSynthesis.speak(
        speech
    );

}


// =========================================================
// START
// =========================================================

document.addEventListener(

    "DOMContentLoaded",

    function () {

        updateAutoMode(
            "waiting"
        );


        updateSequence(
            "waiting",
            0,
            30
        );


        updateStatus();


        setInterval(

            updateStatus,

            300
        );

    }

);