// ==============================================
// GLOBAL
// ==============================================

let currentMode = "static";


// ==============================================
// SET MODE
// ==============================================

async function setMode(mode) {

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

            console.error(
                "Khong the chuyen che do"
            );

            return;
        }


        currentMode = mode;


        updateModeUI();

        await updateStatus();

    }

    catch (error) {

        console.error(
            "Loi set mode:",
            error
        );

    }

}


// ==============================================
// UPDATE MODE UI
// ==============================================

function updateModeUI() {

    const staticButton =
        document.getElementById(
            "staticButton"
        );


    const dynamicButton =
        document.getElementById(
            "dynamicButton"
        );


    const modeText =
        document.getElementById(
            "mode"
        );


    const instruction =
        document.getElementById(
            "cameraInstruction"
        );


    const sequenceBlock =
        document.getElementById(
            "sequenceBlock"
        );


    const spaceButton =
        document.getElementById(
            "spaceButton"
        );


    // reset active
    staticButton
        .classList
        .remove("active");


    dynamicButton
        .classList
        .remove("active");


    // ==========================================
    // STATIC
    // ==========================================

    if (currentMode === "static") {

        staticButton
            .classList
            .add("active");


        modeText.innerText =
            "Chữ cái tĩnh";


        instruction.innerText =
            "Đưa tay vào camera để nhận diện chữ cái";


        sequenceBlock
            .classList
            .add("hidden");


        spaceButton.style.display =
            "inline-block";

    }

    // ==========================================
    // DYNAMIC
    // ==========================================

    else {

        dynamicButton
            .classList
            .add("active");


        modeText.innerText =
            "Ký hiệu động";


        instruction.innerText =
            "Thực hiện đầy đủ hành động ký hiệu trước camera";


        sequenceBlock
            .classList
            .remove("hidden");


        // dynamic tự thêm từ có khoảng trắng
        spaceButton.style.display =
            "none";

    }

}


// ==============================================
// STATUS
// ==============================================

async function updateStatus() {

    try {

        const response =
            await fetch(
                "/status"
            );


        const data =
            await response.json();


        // MODE
        currentMode =
            data.mode || "static";


        updateModeUI();


        // ======================================
        // PREDICTION
        // ======================================

        document
            .getElementById(
                "prediction"
            )
            .innerText =
            data.prediction
            || "---";


        // ======================================
        // CONFIRMED
        // ======================================

        document
            .getElementById(
                "confirmed"
            )
            .innerText =
            data.confirmed
            || "---";


        // ======================================
        // CONFIDENCE
        // ======================================

        const confidence =
            Number(
                data.confidence || 0
            );


        document
            .getElementById(
                "confidence"
            )
            .innerText =
            confidence.toFixed(1)
            + "%";


        document
            .getElementById(
                "progressBar"
            )
            .style.width =
            confidence + "%";


        // ======================================
        // SEQUENCE
        // ======================================

        const sequence =
            Number(
                data.sequence || 0
            );


        document
            .getElementById(
                "sequence"
            )
            .innerText =
            sequence + " / 30";


        // ======================================
        // TEXT
        // ======================================

        const text =
            data.text || "";


        document
            .getElementById(
                "textResult"
            )
            .innerText =
            text.trim()
            ? text
            : "Chưa có nội dung";

    }

    catch (error) {

        console.error(
            "Loi update status:",
            error
        );

    }

}


// ==============================================
// ADD RESULT
// ==============================================

async function addResult() {

    try {

        const response =
            await fetch(
                "/add_result",
                {
                    method: "POST"
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            console.log(
                "Chua co ket qua duoc xac nhan"
            );

        }


        await updateStatus();

    }

    catch (error) {

        console.error(
            "Loi add result:",
            error
        );

    }

}


// ==============================================
// ADD SPACE
// ==============================================

async function addSpace() {

    // Chỉ dùng cho chữ cái tĩnh
    if (currentMode !== "static") {
        return;
    }


    try {

        await fetch(
            "/add_space",
            {
                method: "POST"
            }
        );


        await updateStatus();

    }

    catch (error) {

        console.error(
            "Loi add space:",
            error
        );

    }

}


// ==============================================
// DELETE LAST
// ==============================================

async function deleteLast() {

    try {

        await fetch(
            "/delete_last",
            {
                method: "POST"
            }
        );


        await updateStatus();

    }

    catch (error) {

        console.error(
            "Loi delete:",
            error
        );

    }

}


// ==============================================
// CLEAR
// ==============================================

async function clearText() {

    try {

        await fetch(
            "/clear",
            {
                method: "POST"
            }
        );


        // Dừng speech nếu đang đọc
        window
            .speechSynthesis
            .cancel();


        await updateStatus();

    }

    catch (error) {

        console.error(
            "Loi clear:",
            error
        );

    }

}


// ==============================================
// TEXT TO SPEECH
// ==============================================

function speakText() {

    const element =
        document.getElementById(
            "textResult"
        );


    const text =
        element.innerText.trim();


    if (
        !text
        ||
        text === "Chưa có nội dung"
    ) {

        return;

    }


    window
        .speechSynthesis
        .cancel();


    const speech =
        new SpeechSynthesisUtterance(
            text
        );


    speech.lang =
        "vi-VN";


    speech.rate =
        0.9;


    speech.pitch =
        1;


    speech.volume =
        1;


    window
        .speechSynthesis
        .speak(
            speech
        );

}


// ==============================================
// KEYBOARD SHORTCUTS
// ==============================================

document.addEventListener(
    "keydown",
    function (event) {

        // Không xử lý khi đang gõ input
        if (
            event.target.tagName === "INPUT"
            ||
            event.target.tagName === "TEXTAREA"
        ) {

            return;

        }


        // ENTER = thêm kết quả
        if (
            event.key === "Enter"
        ) {

            addResult();

        }


        // SPACE = khoảng trắng
        else if (
            event.code === "Space"
            &&
            currentMode === "static"
        ) {

            event.preventDefault();

            addSpace();

        }


        // BACKSPACE
        else if (
            event.key === "Backspace"
        ) {

            event.preventDefault();

            deleteLast();

        }


        // ESC = clear
        else if (
            event.key === "Escape"
        ) {

            clearText();

        }

    }
);


// ==============================================
// INITIALIZE
// ==============================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        updateModeUI();

        updateStatus();


        // update 2 lần / giây
        setInterval(
            updateStatus,
            500
        );

    }
);