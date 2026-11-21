import { createBrowserRouter, Navigate } from 'react-router-dom'
import { WorkspaceLayout } from '../layouts/WorkspaceLayout'
import { HelpFeedbackPage } from '../pages/HelpFeedbackPage'
import { SoundMonitorPage } from '../pages/SoundMonitorPage'

export const router = createBrowserRouter([
  {
    element: <WorkspaceLayout />,
    children: [
      { index: true, element: <SoundMonitorPage /> },
      { path: 'monitor', element: <SoundMonitorPage /> },
      { path: 'help', element: <HelpFeedbackPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])
