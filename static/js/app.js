// =========================================================
// GLOBAL
// =========================================================

let currentMode = "static";

let changingMode = false;


// =========================================================
// SET MODE
// =========================================================

async function setMode(mode) {

    if (changingMode) {
        return;
    }


    if (
        mode !== "static"
        &&
        mode !== "dynamic"
    ) {

        return;
    }


    changingMode = true;


    try {

        const response = await fetch(

            "/set_mode",

            {
                method: "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body: JSON.stringify({

                    mode: mode

                })
            }
        );


        const data =
            await response.json();


        if (!data.success) {

            console.log(
                "Khong chuyen duoc mode."
            );

            return;
        }


        currentMode = mode;


        updateModeUI(
            mode
        );


        resetRecognitionUI();


    } catch (error) {

        console.error(
            error
        );

    } finally {

        changingMode = false;
    }

}


// =========================================================
// UPDATE MODE UI
// =========================================================

function updateModeUI(mode) {

    const staticBtn =
        document.getElementById(
            "staticBtn"
        );


    const dynamicBtn =
        document.getElementById(
            "dynamicBtn"
        );


    const currentModeElement =
        document.getElementById(
            "currentMode"
        );


    const description =
        document.getElementById(
            "modeDescription"
        );


    const modelName =
        document.getElementById(
            "modelName"
        );


    const sequenceContainer =
        document.getElementById(
            "sequenceContainer"
        );


    // =====================================================
    // DYNAMIC
    // =====================================================

    if (mode === "dynamic") {

        staticBtn.classList.remove(
            "active"
        );


        dynamicBtn.classList.add(
            "active"
        );


        currentModeElement.innerText =
            "Từ giao tiếp";


        modelName.innerText =
            "LSTM";


        description.innerText =
            "Nhận diện từ/cụm từ động như Xin chào, Xin lỗi từ chuỗi 30 frame.";


        sequenceContainer.style.display =
            "block";

    }

    // =====================================================
    // STATIC
    // =====================================================

    else {

        dynamicBtn.classList.remove(
            "active"
        );


        staticBtn.classList.add(
            "active"
        );


        currentModeElement.innerText =
            "Đánh vần chữ cái";


        modelName.innerText =
            "Random Forest";


        description.innerText =
            "Nhận diện các chữ cái A-Z để đánh vần tên người và tên địa điểm.";


        sequenceContainer.style.display =
            "none";

    }

}


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

        if (
            data.mode
            &&
            data.mode !== currentMode
        ) {

            currentMode =
                data.mode;


            updateModeUI(
                currentMode
            );

        }


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
        // TEXT
        // =================================================

        updateText(
            data.text || ""
        );


        // =================================================
        // DYNAMIC SEQUENCE
        // =================================================

        if (
            data.mode
            === "dynamic"
        ) {

            const current =
                Number(
                    data.sequence || 0
                );


            const total =
                Number(
                    data.sequence_length
                    || 30
                );


            document.getElementById(
                "sequence"
            ).innerText =

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


            percent =
                Math.min(
                    percent,
                    100
                );


            document.getElementById(
                "sequenceBar"
            ).style.width =

                percent + "%";

        }


        // =================================================
        // STATUS
        // =================================================

        updateRecognitionStatus(

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
// STATUS MESSAGE
// =========================================================

function updateRecognitionStatus(
    waiting,
    sequence,
    sequenceLength
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
            "Đã nhận diện - hãy bỏ tay ra để tiếp tục";


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

    if (
        currentMode
        === "static"
    ) {

        element.innerText =
            "Sẵn sàng nhận diện chữ cái";


        return;
    }


    // =====================================================
    // DYNAMIC
    // =====================================================

    if (
        sequence
        < sequenceLength
    ) {

        element.innerText =

            "Đang thu chuyển động "
            + sequence
            + "/"
            + sequenceLength;

    } else {

        element.innerText =
            "Đang nhận diện từ...";
    }

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
// DELETE LAST
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

            updateText(
                ""
            );


            resetRecognitionUI();
        }


    } catch (error) {

        console.error(
            error
        );
    }

}


// =========================================================
// RESET UI
// =========================================================

function resetRecognitionUI() {

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
        "sequence"
    ).innerText = "0/30";


    document.getElementById(
        "sequenceBar"
    ).style.width = "0%";


    updateRecognitionStatus(

        false,

        0,

        30
    );

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
            "Trình duyệt không hỗ trợ Text-to-Speech."
        );

        return;
    }


    // Dừng giọng trước
    window.speechSynthesis.cancel();


    const speech =
        new SpeechSynthesisUtterance(
            text
        );


    speech.lang =
        "vi-VN";


    speech.rate =
        1;


    speech.pitch =
        1;


    speech.volume =
        1;


    window.speechSynthesis.speak(
        speech
    );

}


// =========================================================
// INITIALIZE
// =========================================================

document.addEventListener(

    "DOMContentLoaded",

    function () {

        updateModeUI(
            "static"
        );


        updateStatus();


        // cập nhật khoảng
        // 3 lần / giây

        setInterval(

            updateStatus,

            300

        );

    }

);