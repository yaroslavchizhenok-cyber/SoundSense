import { createBrowserRouter, Navigate } from 'react-router-dom'
import { WorkspaceLayout } from '../layouts/WorkspaceLayout'
import { ActivityPage } from '../pages/ActivityPage'
import { HelpFeedbackPage } from '../pages/HelpFeedbackPage'
import { SoundMonitorPage } from '../pages/SoundMonitorPage'

export const router = createBrowserRouter([
  {
    element: <WorkspaceLayout />,
    children: [
      { index: true, element: <SoundMonitorPage /> },
      { path: 'monitor', element: <SoundMonitorPage /> },
      { path: 'activity', element: <ActivityPage /> },
      { path: 'help', element: <HelpFeedbackPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])
