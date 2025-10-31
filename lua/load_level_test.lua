-- =============================================================================
-- Fire 'n Ice - Level Loader Debugging Script for FCEUX (v2 - Robust)
-- =============================================================================
--
-- INSTRUCTIONS:
-- 1. Make sure you have enabled write permissions:
--    Config -> Lua -> Check "Enable 'write' functions..."
-- 2. Run this script, change TARGET_WORLD/TARGET_LEVEL, and press 'L' in the game.
--
-- =============================================================================

-- ==========================================================
-- CONFIGURATION
-- ==========================================================
local TARGET_WORLD = 8 
local TARGET_LEVEL = 8
-- ==========================================================

local SCRIPT_ENABLED = true

function force_load_level(world_num, level_num)
    print("---------------------------------")
    print(string.format("ATTEMPTING LEVEL LOAD: W%d-L%d", world_num, level_num))

    -- Disable the script's main loop to prevent it from running again
    -- until we manually re-enable it.
    SCRIPT_ENABLED = false

    local world_idx = world_num - 1
    local level_idx = level_num - 1

    -- Step 1: Write world and level to RAM
    memory.writebyte(0x00B4, 10*world_idx+level_idx)
    print("RAM values for level/world have been set.")


    -- Step 2: Set the game state flag
    print("Setting game state flag to 6 at $0002...")
    memory.writebyte(0x0002, 0x06)
    print("Game state flag set.")
end

-- Main loop to be run by FCEUX
-- TODO: In real script this needs to work with bridge and not need restarting
local function main()
    -- Only run the display and input logic if the script is active
    if not SCRIPT_ENABLED then
        gui.text(5, 5, "Level load triggered. Restart script to run again.")
        return
    end

    gui.text(5, 5, string.format("Level Loader Active: Target W%d-L%d", TARGET_WORLD, TARGET_LEVEL))
    gui.text(5, 15, "Press 'L' to trigger level load")

    local inputs = input.get()
    if inputs["L"] then
        force_load_level(TARGET_WORLD, TARGET_LEVEL)
    end
end

emu.registerafter(main)

print("Level loader script running.")
print("Ensure write permissions are enabled in Config -> Lua.")
