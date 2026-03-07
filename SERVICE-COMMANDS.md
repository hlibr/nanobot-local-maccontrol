Useful Commands

Check Status:

bash
launchctl list | grep nanobot
(You should see a numeric ID in the first column — that means it's running.)

Stop Manually:

bash
launchctl unload ~/Library/LaunchAgents/com.hlibr.nanobot.plist

Restart Manually:

bash
launchctl unload ~/Library/LaunchAgents/com.hlibr.nanobot.plist &&launchctl load ~/Library/LaunchAgents/com.hlibr.nanobot.plist

Tail Logs:

bash
tail -f ~/.nanobot/logs/gateway.out.log ~/.nanobot/logs/gateway.err.log